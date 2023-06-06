import os
import flask
from PIL import Image
from tensorflow import keras
from flask import Flask , render_template  , request , send_file,  jsonify

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = keras.models.load_model(os.path.join(BASE_DIR , 'model.h5'))

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

def load_image(filename):
    img = keras.preprocessing.image.load_img(filename, target_size=(128, 128))
    img = keras.preprocessing.image.img_to_array(img)
    img = img.reshape(-1, 128, 128, 3)
    img = img.astype('float32')
    img = img / 255.0
    return img

def predictfunc(filename, model):
    # Load the Image
    img = load_image(filename)
    # Load Model
    # Predict the Class/Label
    result = model.predict(img)
    class_index = result.argmax(axis=-1)[0] # get the index of the max value in the result array

    # DEFAULT_IMG = "https://thumbs.dreamstime.com/b/pink-orchid-19433470.jpg"

    if class_index == 0:
        genus =  "Cattleya"
        family = "Orchidaceae"
        description = "Bunga Cattleya, yang termasuk dalam keluarga Orchidaceae, adalah sejenis bunga eksotis yang terkenal karena keindahannya. Genus Cattleya ini memiliki bermacam-macam spesies yang menarik perhatian dengan warna-warna yang mencolok dan bentuk bunga yang elegan. Bunga Cattleya sering dianggap sebagai simbol keindahan dan keanggunan dalam dunia floristik, dan sering digunakan dalam berbagai acara istimewa seperti pernikahan, perayaan, atau sebagai hadiah yang berkesan. Dengan keunikan dan daya tariknya, bunga Cattleya memancarkan pesona yang tak terlupakan."
        imageUrl = "https://i.imgur.com/m4tvW70.jpg"

    elif class_index == 1:
        genus =  "Dendrobium"
        family = "Orchidaceae"
        description = "Dendrobium adalah genus tumbuhan berbunga yang termasuk dalam keluarga Orchidaceae. Dikenal karena keindahan yang memukau dan beragamnya warna yang dimilikinya, anggrek Dendrobium sangat populer di kalangan penggemar bunga dan kolektor. Bunga-bunga yang menakjubkan ini menampilkan pola yang rumit dan kelopak yang halus, menciptakan tampilan yang mempesona. Dengan penampilan yang anggun dan harum yang memikat, anggrek Dendrobium adalah keajaiban alam yang sejati, memikat indra dan menambah sentuhan elegan di setiap lingkungan. Baik digunakan sebagai elemen dekoratif dalam rangkaian bunga atau ditampilkan sebagai tanaman pot, anggrek Dendrobium dihargai karena keanekaragaman yang luar biasa dan daya tariknya yang abadi."
        imageUrl = "https://i.imgur.com/hU9mgBx.jpg"

    elif class_index == 2:
        genus =  "Oncidium"
        family = "Orchidaceae"
        description = "Oncidium, sebuah genus yang termasuk dalam keluarga Orchidaceae, adalah kelompok anggrek yang beragam dan menarik. Dikenal karena warna-warni yang mencolok dan pola yang rumit, Oncidium dihargai karena bunga-bunga yang anggun dan eksotis. Dengan beragam varietas dan kemungkinan hibridisasi, anggrek-anggrek ini menawarkan berbagai bentuk, ukuran, dan aroma, menjadikannya favorit di kalangan penggemar bunga. Mulai dari tandan bunga yang menjuntai dengan lembut hingga tampilan yang mencolok dan berani, Oncidium memberikan sentuhan keindahan tropis pada taman atau rangkaian bunga apa pun."
        imageUrl = "https://i.imgur.com/1ieNqku.jpg"

    elif class_index == 3:
        genus =  "Phalaenopsis"
        family = "Orchidaceae"
        description = "Bunga Phalaenopsis, yang juga dikenal sebagai anggrek kupu-kupu, adalah anggota keluarga Orchidaceae. Bunga yang cantik ini terkenal karena keindahan dan keanggunannya yang memikat hati. Genus Phalaenopsis memiliki sekitar 60 hingga 70 spesies yang tersebar di berbagai wilayah Asia Tenggara, termasuk Indonesia. Bunga ini memiliki kelopak yang lebar dan rata, serta seringkali memiliki warna-warna yang menarik dan pola-pola menawan. Phalaenopsis juga terkenal sebagai bunga hias yang populer di berbagai acara dan sebagai tanaman hias dalam ruangan yang menambah keindahan dan kesegaran di sekitar kita."
        imageUrl = "https://i.imgur.com/RZcLdRI.jpg"

    elif class_index == 4:
        genus =  "Vanda"
        family = "Orchidaceae"
        description = "Vanda, sebuah genus yang termasuk dalam keluarga Orchidaceae, adalah kelompok anggrek yang menakjubkan dengan keindahan yang mempesona dan warna yang mencolok. Bunga-bunga indah ini terkenal dengan kuncup yang besar dan menarik perhatian, datang dalam berbagai warna yang meliputi ungu, merah muda, kuning, dan putih. Dengan pola yang elegan dan rumit, anggrek Vanda berhasil menarik perhatian para pengagum dan pecinta tanaman. Bunga-bunga tropis ini tumbuh subur di lingkungan yang hangat dan lembap, dan akar udara serta daun tebal mereka berkontribusi pada daya tahan dan adaptabilitas mereka. Anggrek Vanda sangat dihargai karena mekar yang tahan lama dan sering dipamerkan di taman, rangkaian bunga, dan acara-acara istimewa, memberikan sentuhan elegansi dan kesan yang istimewa pada setiap tempat."
        imageUrl = "https://i.imgur.com/jEUqOXO.jpg"

    data = {
        'genus' : genus,
        'family' : family,
        'description' : description,
        'imageUrl' : imageUrl,
    }

    return data

@app.route('/')
def home():
        return "flowerLab AI predict API"

@app.route('/predict-image' , methods = ['POST'])
def predictImage():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    error = ''
    target_img = os.path.join(os.getcwd() , 'uploads')

    file = request.files['file']
    if file and allowed_file(file.filename):
        file.save(os.path.join(target_img , file.filename))
        img_path = os.path.join(target_img , file.filename)
        img = file.filename

        res_pred = predictfunc(img_path , model)

    data = {
        'flower_data' : res_pred,
        'success' : True,
        'message' : 'Predict successfully',
    }

    if(len(error) == 0):
        return  jsonify(data), 200
    else:
        return 'error', 400

if __name__ == "__main__":
    app.run(debug = True)


