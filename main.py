from flask import Flask, jsonify, request
from flask_jwt import JWT, jwt_required, current_identity
from werkzeug.security import safe_str_cmp
from flask import render_template, url_for
from util import predict
import convnets
import torch
from datetime import timedelta
from dimp import wrap_segmentation
from wrap_rq import async_split
from train import wrap_train
import singleton_socketio
from redis import Redis
from rq import Queue
from threading import Lock
import numpy as np


MODEL_PATH = "models/model-convnetj.pb"
model = convnets.ConvNetJ(num_classes=10)
# print(model)
if torch.cuda.is_available():
    model = model.cuda()
model.train(False)
if os.path.exists(MODEL_PATH):
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()


class User(object):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

    def __str__(self):
        return "User(id='%s')" % self.id


users = [
    User(1, 'android', 'aqtpnh973hcz322wsvmem2kcyce3p4rm'),
    User(2, 'ios', 'd798khvze3kaxn3syzpr8zkp5n7smuvq'),
    User(3, 'web', '3780c7b25e31bc59d82a31aa9e986687')
]

username_table = {u.username: u for u in users}
userid_table = {u.id: u for u in users}


def authenticate(username, password):
    user = username_table.get(username, None)
    if user and safe_str_cmp(user.password.encode('utf-8'), password.encode('utf-8')):
        return user


def identity(payload):
    user_id = payload['identity']
    return userid_table.get(user_id, None)


app = Flask(__name__)
app.debug = True
app.config['SECRET_KEY'] = 'r8spx57ytcqq3gnj38wr6nd8eqz6q967'
app.config['JWT_EXPIRATION_DELTA'] = timedelta(seconds=1800)

jwt = JWT(app, authenticate, identity)

socketio = singleton_socketio.init_socket(app)
redis_conn = Redis()
q = Queue(connection=redis_conn)
thread_lock = Lock()
thread = None


@app.route('/api/v1/predict', methods=['POST'])
@jwt_required()
def clf():
    request_json = request.get_json()

    if request_json.get("image"):
        try:
            results, confidence_score = predict(model, request_json.get("image"))
        except Exception as error:
            return jsonify({
                "description": repr(error),
                "error": "Bad Request",
                "status_code": "401"
            })
        return jsonify({
            "results": results,
            "confidence": confidence_score,
            "status_code": "200"
        })
    else:
        return jsonify({
            "description": "Input Image is Missing",
            "error": "Bad Request",
            "status_code": "401"
        })


@app.route('/api/v1/segment', methods=['POST'])
@jwt_required()
def segment():
    request_json = request.get_json()

    im64_str = request_json.get("image")
    x1 = request_json.get("x1")
    y1 = request_json.get("y1")
    x2 = request_json.get("x2")
    y2 = request_json.get("y2")
    ean = request_json.get("ean")

    if not im64_str:
        return jsonify({
            "description": "Input Image is Missing",
            "error": "Bad Request",
            "status_code": "401"
        })

    if not x1 or not x2 or not y1 or not y2:
        return jsonify({
            "description": "Rectangle Box is Missing",
            "error": "Bad Request",
            "status_code": "401"
        })

    if not ean:
        return jsonify({
            "description": "EAN is Missing",
            "error": "Bad Request",
            "status_code": "401"
        })

    try:
        box = (int(x1), int(y1), int(x2), int(y2))
        b64_im_res = wrap_segmentation(im64_str, box, ean)
    except Exception as error:
        return jsonify({
            "description": repr(error),
            "error": "Bad Request",
            "status_code": "401"
        })
    return jsonify({
        "output_image": b64_im_res,
        "status_code": "200"
    })


@app.route('/api/v1/generate', methods=['POST'])
@jwt_required()
def generate():
    request_json = request.get_json()

    bg_cardinality = int(request_json.get("number_backgrounds"))
    training_transforms = request_json.get("training_transforms")
    testing_transforms = request_json.get("testing_transforms")
    scales = [int(x) for x in request_json.get("scales")]
    flip_ver = bool(int(request_json.get("flip_vertically")))
    flip_hor = bool(int(request_json.get("flip_horizontally")))
    flip_both = bool(int(request_json.get("flip_both")))
    angels = [int(x) for x in request_json.get("angels")]
    test_angels = [int(x) for x in request_json.get("test_angels")]
    grids = bool(int(request_json.get("apply_occulsion")))
    grid_width = int(request_json.get("occulsion_window"))
    grid_stride = int(request_json.get("occulsion_stride"))
    bool_center_crop = bool(int(request_json.get("apply_center_crop")))
    center_crop_windows = [int(x) for x in request_json.get("center_crop_windows")]

    for trans in training_transforms:
        if trans not in ["gaussian", "contrast", "brightness", "color"]:
            return jsonify({
                "description": "{} is not a valid transformation".format(trans),
                "error": "Bad Request",
                "status_code": "401"
            })

    for trans in testing_transforms:
        if trans not in ["gaussian", "contrast", "brightness", "color"]:
            return jsonify({
                "description": "{} is not a valid transformation".format(trans),
                "error": "Bad Request",
                "status_code": "401"
            })

    for scale in scales:
        if int(scale) <= 0 or int(scale) >= 512:
            return jsonify({
                "description": "{} is not a valid scale".format(scale),
                "error": "Bad Request",
                "status_code": "401"
            })

    for win in center_crop_windows:
        if int(win) <= 0 or int(win) >= 512:
            return jsonify({
                "description": "{} is not a valid crop window".format(win),
                "error": "Bad Request",
                "status_code": "401"
            })

    try:
        jobs = async_split(
            q=q,
            bg_cardinality=bg_cardinality,
            training_transforms=training_transforms,
            testing_transforms=testing_transforms,
            scales=scales,
            flip_ver=flip_ver,
            flip_hor=flip_hor,
            flip_both=flip_both,
            angels=angels,
            grids=grids,
            grid_width=grid_width,
            grid_stride=grid_stride,
            bool_center_crop=bool_center_crop,
            center_crop_windows=center_crop_windows,
            test_angels=test_angels
        )
    except Exception as error:
        return jsonify({
            "description": repr(error),
            "error": "Bad Request",
            "status_code": "401"
        })
    return jsonify({
        "jobs": jobs,
        "status_code": "200"
    })


@app.route('/api/v1/train', methods=['POST'])
@jwt_required()
def train():
    request_json = request.get_json()

    model_name = request_json.get("model_name")
    dataset = request_json.get("dataset")
    num_classes = int(request_json.get("num_classes"))
    batch_size = int(request_json.get("batch_size"))
    is_transform = bool(int(request_json.get("is_transform")))
    num_workers = int(request_json.get("num_workers"))
    lr_decay = bool(int(request_json.get("lr_decay")))
    l2_reg = float(request_json.get("l2_reg"))
    hdf5_path = request_json.get("hdf5_path")
    trainset_dir = request_json.get("trainset_dir")
    testset_dir = request_json.get("testset_dir")
    convert_grey = bool(int(request_json.get("convert_grey")))
    learning_rate = float(request_json.get("learning_rate"))
    num_epochs = int(request_json.get("num_epochs"))

    if model_name not in [
        "alexnet",
        "lenet5",
        "stn-alexnet",
        "stn-lenet5",
        "capsnet",
        "convneta",
        "convnetb",
        "convnetc",
        "convnetd",
        "convnete",
        "convnetf",
        "convnetg",
        "convneth",
        "convneti",
        "convnetj",
        "convnetk",
        "convnetl",
        "convnetm",
        "convnetn",
        "resnet18"
    ]:
        return jsonify({
            "description": "{} is not a valid model".format(model_name),
            "error": "Bad Request",
            "status_code": "401"
        })

    if dataset not in ["custom", "cifar", "hdf5"]:
        return jsonify({
            "description": "{} is not a valid dataset".format(dataset),
            "error": "Bad Request",
            "status_code": "401"
        })

    if num_classes <= 0 or batch_size <= 0 or num_workers <= 0 \
            or num_epochs <= 0:
        return jsonify({
            "description": "num_classes, batch_size, num_workers and num_epochs"
                           " must be greater than zero",
            "error": "Bad Request",
            "status_code": "401"
        })

    try:
        jobs = wrap_train(
            q=q,
            model_name=model_name,
            dataset=dataset,
            num_classes=num_classes,
            batch_size=batch_size,
            is_transform=is_transform,
            num_workers=num_workers,
            lr_decay=lr_decay,
            l2_reg=l2_reg,
            hdf5_path=hdf5_path,
            trainset_dir=trainset_dir,
            testset_dir=testset_dir,
            convert_grey=convert_grey,
            learning_rate=learning_rate,
            num_epochs=num_epochs
        )
    except Exception as error:
        return jsonify({
            "description": repr(error),
            "error": "Bad Request",
            "status_code": "401"
        })
    return jsonify({
        "jobs": jobs,
        "status_code": "200"
    })


@socketio.on('connect', namespace='/test')
def test_connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(
                singleton_socketio.background_thread,
                q,
                redis_conn
            )
    singleton_socketio.push('my_response',
                            {
                                'data': 'Connected',
                                'count': 0
                            },
                            group='/test'
                            )


@app.route('/api/v1/jobs/<jid>', methods=['POST'])
@jwt_required()
def jobs(jid):
    if not jid:
        return jsonify({
            "description": "Job ID is Missing",
            "error": "Bad Request",
            "status_code": "401"
        })

    try:
        with open("{}.npy".format(jid)) as f:
            train_loss = []
            test_loss = []
            train_acc = []
            test_acc = []
            for log in f.readlines():
                vals = log.split(",")
                train_loss.append(float(vals[0]))
                test_loss.append(float(vals[2]))
                train_acc.append(float(vals[1]))
                test_acc.append(float(vals[3]))

    except Exception as error:
        return jsonify({
            "description": repr(error),
            "error": "Bad Request",
            "status_code": "401"
        })
    return jsonify({
        "training_loss": train_loss,
        "testing_loss": test_loss,
        "training_accuracy": train_acc,
        "testing_accuracy": test_acc,
        "status_code": "200"
    })


if __name__ == '__main__':
    # app.run(debug=True)
    socketio.run(app)
