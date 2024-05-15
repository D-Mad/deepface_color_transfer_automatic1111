import modules.scripts as scripts
import gradio as gr
from deepface import DeepFace
import glob
import numpy as np

from modules import script_callbacks

def verify_images(source_image, target_image, model_name, detector, metric, e_detection, alignment):
    # Convert PIL Image to numpy array
    source_image = np.array(source_image)
    target_image = np.array(target_image)

    result = DeepFace.verify(source_image, target_image, model_name, detector_backend=detector, distance_metric=metric, enforce_detection=e_detection, align=alignment)
    useful_result = "Verified:    " + str(result['verified']) + "\nDistance:     " + str(result['distance']) + "\nThreshold:   " + str(result['threshold']) + "\nModel:    " + str(result['model']) + "\nSimilarity metric:   " + str(result['similarity_metric']) + "\nTime:   " + str(result['time'])

    return useful_result

def analysis_images(img, act, detector, e_detection, alignment, silent):
    # Convert PIL Image to numpy array
    img = np.array(img)

    result = DeepFace.analyze(img, act, e_detection, detector, alignment, silent)
    result = result[0]
    output_string = ""
    if 'dominant_emotion' in result:
        emotion = result['dominant_emotion']
        output_string += "Emotion: " + str(emotion) + "\n"
    if 'face_confidence' in result:
        face_confidence = result['face_confidence']
        output_string += "Face confidence: " + str(face_confidence) + "\n"
    if 'age' in result:
        age = result['age']
        output_string += "Age: " + str(age) + "\n"
    if 'dominant_gender' in result:
        gender = result['dominant_gender']
        output_string += "Gender: " + str(gender) + "\n"
    if 'dominant_race' in result:
        region = result['dominant_race']
        output_string += "Region: " + str(region)

    return output_string

def get_glob(path):
    png_img = glob.glob(path + "/*.png")
    jpg_img = glob.glob(path + "/*.jpg")
    return png_img + jpg_img

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as face_recognition_interface:
        with gr.Tab("Verify image"):
            with gr.Row(equal_height=False):
                with gr.Column(variant='panel'):
                    with gr.Tabs(elem_id="image1"):
                        with gr.TabItem('Image 1'):
                            with gr.Row():
                                source_image = gr.Image(label="Image 1", sources=["upload", 'clipboard'], elem_id="img", type="pil")

                    with gr.Tabs(elem_id="image2"):
                        with gr.TabItem('Image 2'):
                            with gr.Row():
                                target_image = gr.Image(label="Image 2", sources=["upload", 'clipboard'], type="pil")
                with gr.Column(variant='panel'):
                    with gr.Tabs(elem_id="checkbox"):
                        with gr.TabItem('Settings'):
                            with gr.Column(variant='panel'):
                                with gr.Row():
                                    model_name = gr.Radio(['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace', 'GhostFaceNet'], value='Facenet512', label='Model for face recognition: ')
                                with gr.Row():
                                    detector = gr.Radio(['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yunet', 'fastmtcnn'], value='retinaface', label='Model for face detection: ')
                                with gr.Row():
                                    metric = gr.Radio(['cosine', 'euclidean', 'euclidean_l2'], value='euclidean_l2', label='Distance metric: ')
                                with gr.Row():
                                    e_detection = gr.Checkbox(label="Enforce detection", value=True, info='If no face could not be detected in an image, then this function will return exception by default. Set this to False not to have this exception. This might be convenient for low resolution images.')
                                with gr.Row():
                                    alignment = gr.Checkbox(label="Alignment according to the eye positions", value=True)

                with gr.Tabs(elem_id="genearted"):
                    with gr.TabItem('Results'):
                        with gr.Column(variant='panel'):
                            result = gr.Textbox(label="Output: ", autofocus=True)
                            submit = gr.Button('Submit', elem_id="submit", variant='primary')
            submit.click(
                fn=verify_images,
                inputs=[
                    source_image,
                    target_image,
                    model_name,
                    detector,
                    metric,
                    e_detection,
                    alignment
                ],
                outputs=[result]
            )
        with gr.Tab("Analysis image"):
            with gr.Row(equal_height=False):
                with gr.Column(variant='panel'):
                    with gr.Tabs(elem_id="image"):
                        with gr.TabItem('Image'):
                            with gr.Row():
                                img = gr.Image(label="Image", sources=["upload", 'clipboard'], elem_id="img", type="pil")
                            with gr.Row():
                                act = gr.CheckboxGroup(["emotion", "age", "gender", "race"], label='Choose attribute to analysis')
                with gr.Column(variant='panel'):
                    with gr.Tabs(elem_id="checkbox"):
                        with gr.TabItem('Settings'):
                            with gr.Column(variant='panel'):
                                with gr.Row():
                                    e_detection = gr.Checkbox(label="Enforce detection", value=True, info='If no face could not be detected in an image, then this function will return exception by default. Set this to False not to have this exception. This might be convenient for low resolution images.')
                                with gr.Row():
                                    detector = gr.Radio(['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yunet', 'fastmtcnn'], value='retinaface', label='Face detector model: ', info="Model for face detection")
                                with gr.Row():
                                    alignment = gr.Checkbox(label="Align", value=True, info='alignment according to the eye positions')
                                with gr.Row():
                                    silent = gr.Checkbox(label="Silent", value=False, info='disable (some) log messages')

                with gr.Tabs(elem_id="genearted"):
                    with gr.TabItem('Results'):
                        with gr.Column(variant='panel'):
                            result = gr.Textbox(label="Output: ", autofocus=True)
                            submit = gr.Button('Submit', elem_id="submit", variant='primary')
            submit.click(
                fn=analysis_images,
                inputs=[
                    img,
                    act,
                    detector,
                    e_detection,
                    alignment,
                    silent
                ],
                outputs=[result]
            )
        with gr.Blocks("Image from path"):
            with gr.Row(variant='panel'):
                with gr.Column(scale=2):
                    gallery = gr.Gallery(elem_id="gallery", height="auto", scale=2)
                with gr.Column():
                    img_path = gr.Textbox(label="Image from path", scale=1)
                    gen = gr.Button("Read Image", elem_id="Read", variant='primary')
                    gen.click(
                        fn=get_glob,
                        inputs=[img_path],
                        outputs=[gallery]
                    )

    return [(face_recognition_interface, "Face Recognition", "face_recognition_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)