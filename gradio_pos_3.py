import gradio as gr
import cv2
from tempfile import NamedTemporaryFile
import numpy as np
from numpy import asarray

#global
#height = 10
#height = []
#height = 600
temp_file_location = None




# Function to extract frame from video
def extract_frame(video_file, frame_number):
    cap = cv2.VideoCapture(video_file)
    
    #da kreiram globalnu 
    global height#, width
    #height = height+5
    height = int(cap.get(4))
    #width = int(cap.get(3))
    print(height)
    #data.append[height]



    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)  # Set the frame number to extract (zero-indexed)
    ret, frame = cap.read()
    cap.release()
    

    
    # Save the frame temporarily
    temp_file = NamedTemporaryFile(delete=False, suffix='.png') #umesto da mu ja smaisljam ime

    global temp_file_location
    temp_file_location = temp_file.name
    #print(temp_file_location)
    print(temp_file.name)

    ###global 
    #data.append(temp_file)
    
    cv2.imwrite(temp_file.name, frame) #--> to see it in bloc image, MORA da stoji!!!
    
    return temp_file.name, gr.update(maximum=height, value=height//2)
    #Update Slider Maximum: The gr.update method is used to update the slider's maximum value and set a default value (midpoint of the frame height).
    #Dakle, sa gr.update mozes da uzimas lokalne promenljive iz funkcija da ih koristis kasnije!!! Ovde su to, za sada maximum i value parametri.
    #I onda kada dole pozivas ovu funkciju, sa onom naredbom btn.click, navedes inputs=[], pa outputs=[]:
    #froze_btn.click(extract_frame, inputs=[video_org, no_frame], outputs=[frozen_frame,slider]) #--> mora da ima dva inputs(vidi def) i dva outputs (vidi return)
    #E, za slider ce da uzme maximum i value iz gr.update(maximum=height, value=height//2)
#------------------------------------------------------------------------------------------------------------------------------------------------------








#image1 = cv2.imread(data)
#print(height)
print(temp_file_location)


#if temp_file_location is not None:
#    height = cv2.imread(temp_file_location).shape[0]
#else:
#    height = 600



#-------------------------------------------------------------------------------------------------------------------------------------------------
# Function to move line upside down on an image
def draw_line(y,image):
    # Make a copy of the image
    img_with_line = image.copy()
    # Draw the horizontal line
    cv2.line(img_with_line, (0, y), (img_with_line.shape[1], y), (255, 0, 0), 2)
    print(img_with_line.shape[0])
    return img_with_line

# Function to draw a trapezoid on the image
def draw_trapezoid(image, top_left_x, top_left_y, top_right_x, top_right_y, bottom_left_x, bottom_left_y, bottom_right_x, bottom_right_y):
    # Make a copy of the image
    img_with_trapezoid = image.copy()
    
    # Define points of the trapezoid
    points = np.array([
        [top_left_x, top_left_y], 
        [top_right_x, top_right_y], 
        [bottom_right_x, bottom_right_y], 
        [bottom_left_x, bottom_left_y]
    ])
    
    # Draw the trapezoid
    cv2.polylines(img_with_trapezoid, [points], isClosed=True, color=(255, 0, 0), thickness=2)
    
    return img_with_trapezoid
#-------------------------------------------------------------------------------------------------------------------------------------------------





#def dynamic_range_for_slider(height):
#    if height!=600:
#        return height


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            video_org = gr.Video(label="Upload a video file (.mp4, .avi, etc.)")
            no_frame = gr.Number(label="Frame Number (1-indexed)")
            froze_btn = gr.Button(value = 'Extract frame')
        with gr.Column():
            frozen_frame = gr.Image(label="Extracted Frame")#, type="pil") 
            frozen_frame_drawn = gr.Image(label="Modified Image")
            slider = gr.Slider(minimum = 0, maximum = 600, value=50, label="Horizonal line position", info="Choose line position for vehicle counts")#, height//2) #, label="Move Line")
            
            # Sliders to define trapezoid vertices
            top_left_x = gr.Slider(minimum = 0, maximum = 700, value=100, label="Top Left X")
            top_left_y = gr.Slider(minimum = 0, maximum = 700, value=100, label="Top Left Y")
            top_right_x = gr.Slider(minimum = 0, maximum = 700, value=100, label="Top Right X")
            top_right_y = gr.Slider(minimum = 0, maximum = 700, value=100, label="Top Right Y")
            bottom_left_x = gr.Slider(minimum = 0, maximum = 700, value=100, label="Bottom Left X")
            bottom_left_y = gr.Slider(minimum = 0, maximum = 700, value=100, label="Bottom Left Y")
            bottom_right_x = gr.Slider(minimum = 0, maximum = 700, value=100, label="Bottom Right X")
            bottom_right_y = gr.Slider(minimum = 0, maximum = 700, value=100, label="Bottom Right Y")

            # Image output component
            image_output = gr.Image(label="Image with Trapezoid")
            
        
        froze_btn.click(extract_frame, inputs=[video_org, no_frame], outputs=[frozen_frame, slider]) 
        print(type(frozen_frame)) 
        #cv2.imread('C:/Users/N10/AppData/Local/Temp/tmpcfnk3qiq.png').shape[0]
        slider.release(draw_line, inputs=[slider, frozen_frame], outputs=[frozen_frame_drawn])#, api_name="predict")


        print(frozen_frame.width)


        input_t = [frozen_frame, top_left_x, top_left_y, top_right_x, top_right_y, bottom_left_x, bottom_left_y, bottom_right_x, bottom_right_y]
        #gr.Row([gr.update(output=image_output)]).update(draw_trapezoid, inputs=inputs, outputs=image_output)
        #gr.update(draw_trapezoid, inputs=[frozen_frame, top_left_x, top_left_y, top_right_x, top_right_y, bottom_left_x, bottom_left_y, bottom_right_x, bottom_right_y], outputs=image_output)
        top_left_x.release(draw_trapezoid, inputs=input_t, outputs=[image_output])
        top_left_y.release(draw_trapezoid, inputs=input_t, outputs=[image_output])
        top_right_x.release(draw_trapezoid, inputs=input_t, outputs=[image_output])
        top_right_y.release(draw_trapezoid, inputs=input_t, outputs=[image_output])
        bottom_left_x.release(draw_trapezoid, inputs=input_t, outputs=[image_output])
        bottom_left_y.release(draw_trapezoid, inputs=input_t, outputs=[image_output])
        bottom_right_x.release(draw_trapezoid, inputs=input_t, outputs=[image_output])
        bottom_right_y.release(draw_trapezoid, inputs=input_t, outputs=[image_output])
        
demo.launch()
