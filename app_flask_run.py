#!/usr/bin/env python
# encoding: utf-8
'''
@author: chenc
@time: 2019/1/3 4:34 PM
@desc:
'''
from flask import Flask
from flask import Blueprint
import json
from flask import request

import app_config
from app_synthesize import synthesizeBytex

user_data = [
    {
        'id': 1,
        'name': '张三',
        'age': 23
    },
    {
        'id': 2,
        'name': '李四',
        'age': 24
    }
]

api = Blueprint('api', __name__,template_folder='templates',static_folder='static')

@api.route('/tacotron', methods=['POST'])
def tacotron():
    if request.method == 'POST':
        logId = request.form['logId']
        tex = request.form['tex']

        wavPath = synthesizeBytex(logId, tex)

        data = {
            'logId': logId,
            'tex': tex,
            'wavPath':wavPath
        }

    return json.dumps(data, ensure_ascii=False, indent=1)




def main():
    app = Flask(__name__)
    # app.secret_key = "123asdzxc"
    app.permanent_session_lifetime = 60 * 60 * 2
    app.config.from_object(app_config)
    app.register_blueprint(api, url_prefix='/tts')
    app.run(host='0.0.0.0', port=5031, )
    print("flask 启动成功")


if __name__ == '__main__':
    main()
