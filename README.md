# Overview

A collection of scripts to help upstart data augmentation and hyperparameters optimization for PyTorch with RESTful APIs.

Through RESTful APIs one gets flexibility to add desired GUI or Web Interface that visualizes training in realtime.

This library supports asynchronous jobs via Redis Queue and realtime updates via SocketIO

# Initialization

```
$ cd restful-pytorch-automation
$ rq worker
$ export FLASK_APP=main.py
$ flask run
```

# JWT Authentication

All requests require JWT OAuth2.0 token in order to function.

## Request

### Endpoint

```
{{url}}/auth
```

### Header

```
Content-Type	application/json
Accept      	application/json
```

### Body

```
{
    "username":"ios",
    "password":"d798khvze3kaxn3syzpr8zkp5n7smuvq"
}
```

# Grab Cut Segmentation

This method accepts an image and object bounding box coordinates and returns the object after the background has been removed.

This step functions as a preprocessing step for data augmentation. The background should be simple monochrome with shadows avoided.

EAN code is only a placeholder for the class label or category for the classification task.

Images should be base64 encoded.

## Request

### Endpoint

```
{{url}}/api/v1/segment
```

### Header

```
Content-Type	application/json
Accept      	application/json
Authorization   JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE1MjkyNTIwMzIsImlhdCI6MTUyOTI1MDIzMiwibmJmIjoxNTI5MjUwMjMyLCJpZGVudGl0eSI6Mn0.f_qLJwkOulNyz8jRlSQiFukSUnbBF9kSwudcPjbG-jU
```

### Body

```
{
    "x1": "37",
    "x2": "220",
    "y1": "35",
    "y2": "179",
    "ean": "0601472600",
    "image":"data:image/jpeg;base64,/9j/..."
}
```

# Data Augmentation

Produce numerous variations for the same object to make the model robust to transformations:

1- Synthetic backgrounds found under bgs/ directory
2- rotations
3- scale
4- brightness
5- contrast
6- gamma
7- gaussian blur
8- flip
9- crop center
10- crop sliding window

The task is handled asynchronously via Redis Queue RQ and Job ID is returned for tracking

## Request

### Endpoint

```
{{url}}/api/v1/generate
```

### Header

```
Content-Type	application/json
Accept      	application/json
Authorization   JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE1MjkyNTIwMzIsImlhdCI6MTUyOTI1MDIzMiwibmJmIjoxNTI5MjUwMjMyLCJpZGVudGl0eSI6Mn0.f_qLJwkOulNyz8jRlSQiFukSUnbBF9kSwudcPjbG-jU
```

### Body

```
{
    "number_backgrounds": "2",
    "training_transforms": [
        "brightness",
        "color"
    ],
    "testing_transforms": [
        "gaussian",
        "contrast"
    ],
    "scales": [
        "500",
        "490"
    ],
    "flip_vertically": "1",
    "flip_horizontally": "1",
    "flip_both": "1",
    "angels": [
        "15",
        "-15",
        "30",
        "-30"
    ],
    "test_angels":[
        "5",
        "-5",
        "20",
        "-20",
        "35",
        "-35"
    ],
    "apply_occulsion": "0",
    "occulsion_window": "0",
    "occulsion_stride": "0",
    "apply_center_crop": "0",
    "center_crop_windows": [
        112,
        56
    ]
}
```

# Training

Different models exist to select from such as Spatial Transform Network, Capsule Network, AlexNets and custom convolutional networks with different designs.

The task is handled asynchronously via Redis Queue RQ and Job ID is returned for tracking

## Request

### Endpoint

```
{{url}}/api/v1/train
```

### Header

```
Content-Type	application/json
Accept      	application/json
Authorization   JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE1MjkyNTIwMzIsImlhdCI6MTUyOTI1MDIzMiwibmJmIjoxNTI5MjUwMjMyLCJpZGVudGl0eSI6Mn0.f_qLJwkOulNyz8jRlSQiFukSUnbBF9kSwudcPjbG-jU
```

### Body

```
{
	"model_name":"convnetj",
	"dataset":"custom",
	"num_classes":3,
	"batch_size":16,
	"is_transform":1,
	"num_workers":2,
	"lr_decay":1,
	"l2_reg":0.0001,
	"hdf5_path":"dataset-512x512.hdf5",
	"trainset_dir":"TRAIN_data_224",
	"testset_dir":"TEST_data_224",
	"convert_grey":0,
	"learning_rate":0.01,
	"num_epochs":4000
}
```

# Realtime Updates

The values for training loss, training accuracy, testing loss, and testing accuracy are returned using server push for every epoch.

It's possible to use such data to plot realtime graph with JavaScript to visualize the training progress.

```
<script type="text/javascript" src="http://cdnjs.cloudflare.com/ajax/libs/socket.io/1.3.6/socket.io.min.js"></script>
<script type="text/javascript" charset="utf-8">
    var socket = io.connect('http://35.197.120.161:5000/test');

    socket.on('connect', function() {
        socket.emit('my event', {data: 'I\'m connected!'});
    });

    socket.on('my_response', function(msg) {
        console.log(msg);
    });

</script>
```

# Inference / Test

Get predicted class / label for a single image which is encoded using base64.

Note: Update the labels in the source code if necessary

## Request

### Endpoint

```
{{url}}/api/v1/predict
```

### Header

```
Content-Type	application/json
Accept      	application/json
Authorization   JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE1MjkyNTIwMzIsImlhdCI6MTUyOTI1MDIzMiwibmJmIjoxNTI5MjUwMjMyLCJpZGVudGl0eSI6Mn0.f_qLJwkOulNyz8jRlSQiFukSUnbBF9kSwudcPjbG-jU
```

### Body

```
{
	"image":"data:image/jpeg;base64,/9j/..."
}
```