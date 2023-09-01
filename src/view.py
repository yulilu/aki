import pytorch_lightning as pl
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics.functional import accuracy
import torchsummary
from torchsummary import summary
from pytorch_lightning.loggers import CSVLogger
import torch
from flask import Flask, request, render_template, redirect, url_for
import io
import base64
import os
from werkzeug.utils import secure_filename
import openai
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from gtts import gTTS
import uuid
import requests
from transformers import AlbertTokenizer, AlbertForSequenceClassification
import logging

app = Flask(__name__, static_url_path='/static', static_folder='../static')
app.logger.setLevel(logging.DEBUG)

ALLOWED_EXTENSIONS = {'mp3', 'm4a'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def getCategory(response):
    if response == 0:
        return '0_最初の挨拶'
    elif response == 1:
        return '1_繋ぎ確認'
    elif response == 2:
        return '2_要件確認'
    elif response == 3:
        return '3_やりとり有無の確認'
    elif response == 4:
        return '4_現在のやり取り内容の確認'
    elif response == 5:
        return '5_取引有無の確認'
    elif response == 6:
        return '6_営業ですか'
    elif response == 7:
        return '7_前任退社'
    elif response == 8:
        return '8_前任退社_後任知ってますか'
    elif response == 9:
        return '9_前任退社_後任に連絡してください'
    elif response == 10:
        return '10_不在'
    elif response == 11:
        return '11_不在_携帯番号はご存知でしょうか'
    elif response == 12:
        return '12_不在_携帯番号にかけてください'
    elif response == 13:
        return '13_不在_要件確認'
    elif response == 14:
        return '14_不在_やりとり有無の確認'
    elif response == 15:
        return '15_不在_現在のやり取り内容の確認'
    elif response == 16:
        return '16_不在_取引有無の確認'
    elif response == 17:
        return '17_不在_営業ですか'
    elif response == 18:
        return '18_不在_出張中'
    elif response == 19:
        return '19_不在_掛け直し希望'
    elif response == 20:
        return '20_不在_戻り時間共有'
    elif response == 21:
        return '21_受付NG'
    elif response == 22:
        return '22_受付NG_問い合わせフォーム'
    elif response == 23:
        return '23_受付NG_メール送付してください'
    elif response == 24:
        return '24_担当存在しない'
    elif response == 25:
        return '25_担当名確認'
    else:
        return 'Error'

def generatedResponse(response):
    if response == 0:
        return 'お世話'#ここは文字が自然と入るようにしたい
        #return 'お世話になっております。キカガクのフナクラと申します。担当のXX様はいらっしゃりますでしょうか'#ここは文字が自然と入るようにしたい
    elif response == 1:
        return 'ありがとうございます'
    elif response == 2:
        return '弊社はキカガクというAI関連の法人サービスを扱っておりまして、日本国内では最大級の案件数を誇るのですが、昨今の情勢の中でAIに対する温度感は皆様非常に高くなっているので、貴社もかなりこの領域を調べてらっしゃるようでしたのでお電話させていただきました。'
    elif response == 3:
        return 'リンクトインでやりとりさせていただいております'
    elif response == 4:
        return '弊社キカガクというAI関連の法人サービスを扱っておりまして、そちらのAI研修の件です'
    elif response == 5:
        return 'これから契約というタイミングです'
    elif response == 6:
        return 'はい、営業です。我々がキカガクというAI関連の法人サービスを扱っており日本国内では最大級の案件数を誇るのですが、昨今の情勢の中でAIに対する温度感は皆様非常に高くなっているので、貴社もかなりこの領域を調べてらっしゃるようでしたのでお電話させていただきました。'
    elif response == 7:
        return 'そちら、最新のご状況を存じ上げず失礼致しました。後任者様をご共有いただく前にご退職なさったみたいですので、後任の方にお繋ぎいただけますでしょうか'
    elif response == 8:
        return 'そちら、最新のご状況を存じ上げず失礼致しました。後任者様をご共有いただく前にご退職なさったみたいですので、存じ上げずでして。後任の方にお繋ぎいただけますでしょうか'
    elif response == 9:
        return 'そちら、最新のご状況を存じ上げず失礼致しました。連絡先がわからずでして、お手数ですが、後任者の方のお名前とメールアドレスをご教示いただけないでしょうか'
    elif response == 10:
        return '承知しました。お電話口の方に何度もご確認いただくお手間をおかけするのも申し訳ないですので、何時ごろお戻りになられるかご教示いただけないでしょうか？'
    elif response == 11:
        return 'そうだったのですね。すみません、携帯番号をちょっと知らずでして。よろしければ何度もご確認いただくお手間をおかけするのも申し訳ないですので、教えていただけますでしょうか'
    elif response == 12:
        return '承知いたしました。申し訳ないのですが、携帯番号を存じておらずでして。。教えていただくことは可能でしょうか'
    elif response == 13:
        return 'あ、ご不在なんですね。要件としては、我々がキカガクというAI関連の法人サービスを扱っており日本国内では最大級の案件数を誇るのですが、昨今の情勢の中でAIに対する温度感は皆様非常に高くなっており、そのような研修の件でお話ししていた次第です'
    elif response == 14:
        return 'あ、ご不在なんですね。以前担当が名刺交換させていただいたようでして、そのときはやりとりがあったのですが、私になってからは初めてです'
    elif response == 15:
        return 'あ、ご不在なんですね。要件としては、我々がキカガクというAI関連の法人サービスを扱っており日本国内では最大級の案件数を誇るのですが、そちらに関してでございます'
    elif response == 16:
        return 'あ、ご不在なんですね。現在は取引はない形です'
    elif response == 17:
        return 'あ、ご不在なんですね。はい、営業です。我々がキカガクというAI関連の法人サービスを扱っており日本国内では最大級の案件数を誇るのですが、昨今の情勢の中でAIに対する温度感は皆様非常に高くなっているので、お電話させていただきました。'
    elif response == 18:
        return 'あ、ご出張なんですね。ではご出張終わりのタイミングで改めれればと存じるのですが、いつ出張からお戻りでしょうか'
    elif response == 19:
        return 'かしこまりました。何度もお電話口の方にお手間取らせるのは申し訳ないのでお戻りのお時間に私から連絡できればと存じるのですが、何時ごろお戻りでしょう以下'
    elif response == 20:
        return 'かしこまりました。では、その時間にお電話させていただきます'
    elif response == 21:
        return '承知いたしました。またご縁がございましたら、よろしくお願いいたします'
    elif response == 22:
        return '承知いたしました。では問い合わせフォームにてご連絡させていただきますので、そちらのフォームの内容を可能であれば今週中にご確認いただけますようお伝えいただけますでしょうか'
    elif response == 23:
        return '承知いたしました。誠に恐れ入りますが、送付先となるメールアドレスをご教示いただくことは可能でしょうか'
    elif response == 24:
        return '大変失礼いたしました。それでは現在後後任になられてらっしゃる方にお繋ぎいただくことは可能でしょうか'
    elif response == 25:
        return 'すみません、初めましてでお電話させていただいているのですが、我々がキカガクというAI関連の法人サービスを扱っており日本国内では最大級の案件数を誇りまして、昨今のAIの進展からこれを機に研修を企画されている部長様が非常に多くなっておりまして、絶対にガッカリさせないのでお繋ぎいただけますでしょうか'
    else:
        return 'Error'    

model_path = './src/model_files'
tokenizer_path = './src/tokenizer_files'

albert_model = AlbertForSequenceClassification.from_pretrained(model_path).cpu().eval()
albert_tokenizer = AlbertTokenizer.from_pretrained(tokenizer_path)

def predict(text):
    # テキストのエンコード
    input_encodings = albert_tokenizer(
        text,
        return_tensors='pt',
        max_length=70, 
        padding='max_length',
        truncation=True
    )

    # 推論
    with torch.no_grad():
        outputs = albert_model(**input_encodings)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).cpu().numpy()[0]
        
    return predicted_label

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            #wisper AIに食べさせられるように音声ファイルをrb形でopenしておく
            with open(file_path, "rb") as f:
                #wisper AIに音源データを文字起こしさせる
                #transcription = openai.Audio.transcribe("whisper-1", f)
                #txt = transcription['text']
                txt = "This is a pen"
            #ここで取得した音声データの文字データtxtをベースに推論を実施

            from datetime import datetime
            app.logger.debug("現在の日時1s: %s", datetime.now())
            pred = predict(txt)
            app.logger.debug("現在の日時1e: %s", datetime.now())
            category_ = getCategory(pred)
            generatedResponse_ = generatedResponse(pred)
            # テキストを音声に変換。まずは初回はカテゴリーnameを音声として入れる
            app.logger.debug("現在の日時2s: %s", datetime.now())
            tts = gTTS(text=generatedResponse_, lang='ja')
            app.logger.debug("現在の日時2e: %s", datetime.now())
            #ファイル名を動的にユニークに生成→flaskの使用上、音声ファイルはstatic/audioというファイルでやらないとダメ
            file_name = str(uuid.uuid4()) + ".mp3"
            file_path = os.path.join('static/audio', file_name)

            # 音声をmp3ファイルとして保存
            tts.save(file_path)
            return render_template('result.html', audio_file_url=url_for('static', filename='audio/' + file_name), category=category_)
            

    elif request.method == 'GET':
        return render_template('index.html')

if __name__ == '__main__':
    app.debug = True
    app.run()
