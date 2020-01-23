import numpy as np
import io
from io import BytesIO
import base64
from PIL import Image
import cv2
import keras
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import Flask
from flask import jsonify
from flask import request
import tensorflow as tf

K.clear_session()

tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf.keras.backend.set_session(tf.Session(config=config))

app = Flask(__name__)

def get_model():
    global model
    model = load_model('model_raf.h5')
    model.summary()
    model.load_weights('weight_raf.h5')

    global graph
    graph = tf.get_default_graph() 
    
    print('Model loaded!')

def preprocess_image(image, target_size):
    if image.mode != 'L':
        print('convert rgb to grayscale!')
        image = image.convert('L')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image/255.
    # print(image)
    image = np.expand_dims(image, axis=0)
    return image


def grad_cam(image):  
    print(type(image))
    image_for_grad_cam = image.convert('RGB')
    image_for_grad_cam = image_for_grad_cam.resize((100, 100))
    image_for_grad_cam = np.array(image_for_grad_cam)
    print(image_for_grad_cam.shape)
    
    image_for_prediction = image.convert('L')
    image_for_prediction = image_for_prediction.resize((100, 100))
    # image_for_prediction = cv2.cvtColor(np.array(image_for_grad_cam), cv2.COLOR_RGB2GRAY)
    image_for_prediction = np.array(image_for_prediction)
    image_for_prediction = np.expand_dims(image_for_prediction, axis=-1)
    image_for_prediction = np.expand_dims(image_for_prediction, axis=0)
    image_for_prediction = image_for_prediction/255
    

    preds = model.predict(image_for_prediction)
    classes = preds.argmax(axis=-1)
    classes = int(classes)
    
    output = model.output[:, classes]

    last_conv_layer = model.get_layer('concatenate_24')
    grads = K.gradients(output, last_conv_layer.output)[0]

    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    iterate = K.function([model.input], 
                        [pooled_grads, last_conv_layer.output[0]])

    pooled_grads_value, conv_layer_output_value = iterate([image_for_prediction])
    
    for i in range(364):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
 

    heatmap = cv2.resize(heatmap, (100, 100))
    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    hit = np.uint8(heatmap*0.4)
    heatmap_img = cv2.add(hit, image_for_grad_cam)
    
    # heatmap_img = heatmap * 0.4 + image

    # cv2.imshow('d', heatmap_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return heatmap_img
    



print('Loading Keras model...')
get_model()




@app.route('/predict', methods=['POST'])
def predict():
    
    with graph.as_default():      
        message = request.get_json(force=True)
        encoded = message['image']
        decoded = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(decoded))
        processed_image = preprocess_image(image, target_size=(100, 100))  

        grad_cam_image = grad_cam(image)
        print(grad_cam_image.shape)
        print(type(grad_cam_image))




        img_file = Image.fromarray(grad_cam_image)
        b, g, r = img_file.split()
        img_file = Image.merge("RGB", (r, g, b))  
        print(type(img_file))

        # Convert image to base64
        def im_2_base64(image):
            buff = BytesIO()
            image.save(buff, format="png")
            img_str = base64.b64encode(buff.getvalue())
            return img_str

        result = im_2_base64(img_file)
        result = str(result)
        
        





        # encoded_str = base64.b64encode(result)


        # result = str(grad_cam_image)
        # result = base64.b64encode(grad_cam_image)
        # result = str(result)
        # result = Image.fromarray(grad_cam_image)
        
        
        # grad_cam_image = base64.b64encode(grad_cam_image).decode()
        
        # encoded_string = base64.b64encode(grad_cam_image)
        # grad_cam_image = base64.b64encode(grad_cam_image.tobytes())
        # print(type(grad_cam_image))
        # grad_cam_image = base64.b64encode(grad_cam_image).decode()

    

        # grad_cam_image = base64.b64decode(grad_cam_image)
        # grad_cam_image = Image.open(io.BytesIO(grad_cam_image))
        # grad_cam_image = base64.b64encode(grad_cam_image)

        
        






        prediction = model.predict(processed_image).tolist()
      
        response = {
            'prediction':{
                'angry': prediction[0][0],
                'disgusted': prediction[0][1],
                'fearful': prediction[0][2],
                'happy': prediction[0][3],
                'neutral': prediction[0][4],
                'sad': prediction[0][5],
                'surprised': prediction[0][6],
            },
            'grad_cam_image':result
        }

        # print(response)

        return jsonify(response)

