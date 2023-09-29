from flask import Blueprint, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle    
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app_blueprint=Blueprint('app_blueprint',__name__,static_folder="./assets")

@app_blueprint.route('/')
def index():
    return  render_template('index.html')

@app_blueprint.route('/cardpred', methods=['GET', 'POST'])
def cardpred():
    if request.method == 'POST':
        data = request.json
        # print(data['amount'])
        data['Time']=1
        del data['type']
        del data['number']
        del data['zipcode']
        del data['cvc']
        del data['exdate']
        del data['creditsccore']
        loaded_model = pickle.load(open('rf_model.pkl', 'rb'))
        new_data = {
            'Time': [data['Time']],
            'V1': [data['V1']],
            'V2': [data['V2']],
            'V3': [data['V3']],
            'V4': [data['V4']],
            'V5': [data['V5']],
            'V6': [data['V6']],
            'V7': [data['V7']],
            'V8': [data['V8']],
            'V9': [data['V9']],
            'V10': [data['V10']],
            'V11': [data['V11']],
            'V12': [data['V12']],
            'V13': [data['V13']],
            'V14': [data['V14']],
            'V15': [data['V15']],
            'V16': [data['V16']],
            'V17': [data['V17']],
            'V18': [data['V18']],
            'V19': [data['V19']],
            'V20': [data['V20']],
            'V21': [data['V21']],
            'V22': [data['V22']],
            'V23': [data['V23']],
            'V24': [data['V24']],
            'V25': [data['V25']],
            'V26': [data['V26']],
            'V27': [data['V27']],
            'V28': [data['V28']],
            'Amount': [data['Amount']],
        }
        # Convert int64 values to regular Python integers
        for key in new_data:
            new_data[key] = [float(new_data[key][0])]
        new_df = pd.DataFrame(new_data)
        pca=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
                        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 
                        'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
        new_df['MeanV'] = new_df[pca].mean(axis=1)
        new_df['SumV'] = new_df[pca].sum(axis=1)
        new_df['AmountCategory'] = pd.cut(new_df['Amount'], bins=[-1, 50, 200, float('inf')], labels=[1, 2, 3])
        new_df['StdDev'] = new_df[pca].std(axis=1)  
        new_df['Variance'] = new_df[pca].var(axis=1)
        print(new_df)
        # # Make prediction
        prediction = loaded_model.predict(new_df)
        # res={
        #     "stutus":int(prediction[0])
        # }
        # response_data = {'message': 'Data received and processed'}
        pred=int(prediction[0])
        if(pred==1):
            sender_email = 'testemail@gmail.com'
            sender_password = 'password123'
            recipient_email = 'chikunie@example.com'
            subject = 'Card Status'
            body = 'This Transaction has been flagged as a fradulent transaction and will be reported to responsible bodies for further investigation'
            send_email(sender_email, sender_password, recipient_email, subject, body)
        return str(pred)
    else:
        return "cardpred detected"


def send_email(sender_email, sender_password, recipient_email, subject, body, smtp_server='your_smtp_server.com', smtp_port=587):
    try:
        # Create a message
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = recipient_email
        message['Subject'] = subject

        # Add body to the email
        message.attach(MIMEText(body, 'plain'))

        # Establish a connection to the SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Upgrade the connection to a secure TLS connection
        server.login(sender_email, sender_password)

        # Send the email
        text = message.as_string()
        server.sendmail(sender_email, recipient_email, text)

        print('Email sent successfully')
    except Exception as e:
        print(f'Error: {str(e)}')
    finally:
        server.quit()
