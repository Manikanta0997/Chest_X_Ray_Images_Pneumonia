import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
st.header('Go Down For Prediction')
st.write('Pneumonia is a serious respiratory infection caused by bacteria, viruses, fungi, or parasites that can be life-threatening, particularly for vulnerable populations. Detecting pneumonia is crucial for several reasons. Firstly, early diagnosis allows for prompt treatment, reducing complications and improving patient outcomes. Accurate detection is essential in differentiating pneumonia from other respiratory conditions, ensuring proper management. By determining the specific type and severity of pneumonia, detection helps healthcare providers tailor the treatment approach accordingly. Timely identification and treatment also minimize the risk of complications such as pleural effusion and respiratory failure. Additionally, accurate detection plays a vital role in public health surveillance by identifying outbreaks and guiding preventive measures to control the spread of pneumonia. Lastly, appropriate detection of bacterial pneumonia supports antibiotic stewardship efforts, which are crucial in combating antibiotic resistance. Diagnostic methods such as physical examination, chest X-rays, blood tests, sputum culture, and molecular tests aid in the timely and accurate detection of pneumonia. Overall, timely and accurate detection of pneumonia is of paramount importance as it ensures proper treatment, prevents complications, and safeguards public health.')
st.set_option('deprecation.showfileUploaderEncoding', False)

#@st.cache(allow_output_mutation=True)
def load_model():
	model = tf.keras.models.load_model('./Chest_Xray_model_trained.hdf5')
	return model


def predict_class(image, model):

	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [180, 180])

	image = np.expand_dims(image, axis = 0)

	prediction = model.predict(image)

	return prediction


model = load_model()
st.title('Pneumonia Detection')

file = st.file_uploader("Upload an image of a flower", type=["jpg", "png"])


if file is None:
	st.text('Waiting for upload....')

else:
	slot = st.empty()
	slot.text('Running inference....')

	test_image = Image.open(file)

	st.image(test_image, caption="Input Image", width = 400)

	pred = predict_class(np.asarray(test_image), model)

	class_names = ['NORMAL', 'PNEUMONIA']

	result = class_names[np.argmax(pred)]

	output = 'The image is a ' + result

	slot.text('Done')

	st.success(output)

