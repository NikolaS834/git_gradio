#Libraries
#26.7.2024.
#double check if all are used in the project

import gradio as gr
import cv2
from tempfile import NamedTemporaryFile
import numpy as np
from numpy import asarray


# Object Detecion and tracking
from ultralytics import YOLO, solutions
import supervision as sv

#basics
import pandas as pd
from collections import defaultdict, deque





# Choose the model
model = YOLO('yolov8x.pt')
# Objects to detect Yolo
class_IDS = [2, 3, 5, 7] 
myDict = {key: model.model.names[key] for key in class_IDS}  #{2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

#**************** Auxiliary functions *******************************************
# 1. resizing 
def resize_frame(frame, scale_percent):
    """Function to resize an image in a percent scale"""
    width = int(frame.shape[1] * scale_percent / 100)    #frame.shape[1=1920
    height = int(frame.shape[0] * scale_percent / 100)   #frame.shape[0]=1080
    dim = (width, height) #-->tuple

    # resize image
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized

#2. transform trapezoid shape on image into real rectangular (real world case)
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        #ensure float 32 format
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target) #it should be considered like transformation matrix M

    def transform_points(self, points:np.ndarray) -> np.ndarray:
        reshaped_points = points.reshape(-1,1,2).astype(np.float32) #expect point to be defined in 3D space, not on 2D; so we add like empty dummy dimension
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1,2) #now remove this extra dimension (added few lines above)


import tempfile
video_path =''
#3. function to save list of frames into video XVID format
def creating_video_from_frames(output_video_frames):

    # Check if the frames list is not empty
    if len(output_video_frames) == 0:
        print("Error: Empty list of frames.")
        return
    
    ## Create a temporary file to save the video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix="mp4") #--> ovim se kreira fajl, ali prazan PRimer: 'C:\\Users\\N10\\AppData\\Local\\Temp\\tmpjx0iec81.mp4'
    #temp_file = NamedTemporaryFile(delete=False, suffix=".mp4")
    ##temp_file = NamedTemporaryFile(delete=False, suffix='.png') #umesto da mu ja smaisljam ime
    global video_path
    video_path = temp_file.name
    
    
    fps=30

    # Get the dimensions of the first frame
    height = output_video_frames[0].shape[0]
    width = output_video_frames[0].shape[1]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #define output format as XVID, h264
    #fourcc = cv2.VideoWriter_fourcc(*'h264')
    #fourcc = cv2.VideoWriter_fourcc(*'libx264') ->daje error
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height)) #fps is frame per second (here is 24); widt wwould be 640, height 480

    # Write each frame to the video
    for frame in output_video_frames:
        out.write(frame)

    # Release the VideoWriter object
    out.release()


    return video_path



#******************************************************************************************


frame_list = []
#MAIN part

def main_function_vehicles(path):
    #PRIVREMENO  --> posle sa slider-a
    TARGET_WIDTH = 25
    TARGET_HEIGHT = 70

    TARGET = np.array(
        [
            [0, 0],
            [TARGET_WIDTH, 0],
            [TARGET_WIDTH, TARGET_HEIGHT],
            [0, TARGET_HEIGHT]
    ])
    #PRIVREMENO  --> posle sa slider-a
    SOURCE = np.array([[700,220],[1120,220],[1600,637],[55,637]])  





    #14.7.2024.
    # Scaling percentage of original frame (see function above)
    scale_percent = 100 #--> no scaling here

    #To monitor number of particular vehicle types
    veiculos_contador_in = dict.fromkeys(class_IDS, 0)  # veiculos_contador_in = {'2':0, '3':0, '5':0, '7':0}
    veiculos_contador_out = dict.fromkeys(class_IDS, 0) # veiculos_contador_out = {'2':0, '3':0, '5':0, '7':0}

    #saving path
    #output_video_path = 'novo_selo_veki.mp4' #--> save the video file under this name

    #to display name of class of object
    video_info = sv.VideoInfo.from_video_path(path)

    #create instance of bytetrack
    byte_track = sv.ByteTrack(frame_rate = video_info.fps) #fps is frame per second obtained in video_info

    #line for counting
    line_points = [(200,350), (1500,350)] #za roboflow after scaling 50%

    contador_in = 0
    contador_out = 0


    #----------------------------------------------------------------------------------------
    #two smart methods coming from supervision; it uses resolution from video_info to adjust thickness of bbox and text scale
    #1. figure out optimal line thickness
    thickness = sv.calculate_optimal_line_thickness(resolution_wh = video_info.resolution_wh)
    #2. figure out optimal text scale
    text_scale = sv.calculate_optimal_text_scale(resolution_wh = video_info.resolution_wh)

                                                
    #Now, we can use above values                                                
    #1. drawing boxes
    bounding_box_annotator = sv.BoundingBoxAnnotator(
        #thickness = thickness, 
        thickness = 2, 
        color_lookup = sv.ColorLookup.TRACK) #bez ovog zadnjeg, boje su funkcionisale na sledeci nacin: svaka klasa ima svoju boju, npr. car-blue, truck-yellow
                                            #ovim se boje vezuju za id
    #2. label annotator
    label_annotator = sv.LabelAnnotator(
        #text_scale = text_scale, 
        text_scale = 1, 
        #text_thickness = thickness, 
        text_thickness = 2,
        text_position = sv.Position.TOP_CENTER,
        color_lookup = sv.ColorLookup.TRACK) #bez ovog zadnjeg, boje su funkcionisale na sledeci nacin: svaka klasa ima svoju boju, npr. car-blue, truck-yellow
                                            #ovim se boje vezuju za id

    #3. some advance annotator; to display route
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness, 
        trace_length=video_info.fps*2, 
        position=sv.Position.BOTTOM_CENTER,
        color_lookup = sv.ColorLookup.TRACK)#bez ovog zadnjeg, boje su funkcionisale na sledeci nacin: svaka klasa ima svoju boju, npr. car-blue, truck-yellow
                                            #ovim se boje vezuju za id

    #--------------------------------------------------------------


    ##working with polygone zone
    polygon_zone = sv.PolygonZone(SOURCE, frame_resolution_wh = video_info.resolution_wh)

    #create instance of our View Transformer
    view_transformer = ViewTransformer(source = SOURCE, target = TARGET) #--> now go below bytetrack

    #store coordinates of the car in the past (look at the past); storing in python dictionary
    #so, every frame will just add coordinate to this dictionary
    #U svesci videti deque objasnjenje. Za sada, tretiraj to kao listu, ali brzu ( O(1) vs O(n))
    #I'd like to generate a defaultdict that contains a deque. 
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps)) #in our case 30 frames per second; znaci maksimalna duzina je 30 (pmtimo proslost do
                                                                #1 sekunde unazad) Storing coordinates over the last second
    coordinates_counting = defaultdict(lambda: deque(maxlen=2)) # Storing coordinates over the previous frame only; previous + current = 2




    # create frame generator
    frame_generator = sv.get_video_frames_generator(path)
   

    for frame in frame_generator:
        #14.7.2024.
        #Applying resizing of read frame
        frame  = resize_frame(frame, scale_percent)
        
        result = model.predict(frame, conf = 0.5, classes = class_IDS, device = "cpu", verbose = False)[0]
        detections = sv.Detections.from_ultralytics(result) #between detection and tracking introduce selected area; like filtering
        detections = detections[polygon_zone.trigger(detections)] # --> E, ovako. Detektuju se cesto udaljena vozila, pa se gube, pa se javljaju... 
                                                                        #Sve izvan se ne detektuje. Ovo se pise pre byte_track, a nakon detekcije
        detections = byte_track.update_with_detections(detections=detections)

        
        
        
        annotated_frame = frame.copy()
        # Drawing polygon area acting like filtering (detect or not to detect)
        annotated_frame = sv.draw_polygon(annotated_frame, polygon = SOURCE, color = sv.Color.RED)
        # Drawing transition line for in\out vehicles counting 
        cv2.line(annotated_frame, line_points[0], line_points[1], (255,255,0), 8) 
        
        
        
        #below byte track
        #convert bboxes into a list of points
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        
        #Da sada koordinate centra donje linije bbox-ova konvertujem u realan prostor


        
        if points.size!=0: # ---> znaci nije prazno, nesto se detektovalo
            points = view_transformer.transform_points(points=points).astype(int)


        #Now, we want to provide quality labels (part that goes with bbox)
        #looping over tracker id and points
        labels = []
        for tracker_id, class_id, [_,y] in zip(detections.tracker_id, detections.class_id, points):
            coordinates[tracker_id].append(y)
            #bboxing flickering is moving bbox right, left, up, down and can be treated as a noise; in short time interval measures, this uncertanity
            #can cause large error in speed; so we use the next approach --- deque must be at least half full (in our case 15)
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id} {myDict[class_id]}")
            else:
                coordinates_start = coordinates[tracker_id][-1]
                coordinates_end = coordinates[tracker_id][0]
                distance = abs(coordinates_start - coordinates_end)
                time = len(coordinates[tracker_id])/video_info.fps
                speed = distance/time*3.6 #in km/h
                labels.append(f"#{tracker_id} {myDict[class_id]} {int(speed)} km/h")
                
 
      
        #annotated_frame = trace_annotator.annotate(scene = annotated_frame, detections = detections)
        annotated_frame = bounding_box_annotator.annotate(scene = annotated_frame, detections = detections)
        annotated_frame = label_annotator.annotate(scene = annotated_frame, detections = detections, labels = labels) #labels je lista!!!
        annotated_frame = trace_annotator.annotate(scene = annotated_frame, detections = detections)


        #--------------------------------------------  
        # Counting part
        for det in detections:
            trac_id = det[4]
            ymax = det[0][3]   #postion of detection with given id in current frame   ---> compare with previosu frame
            coordinates_counting[trac_id].append(ymax) #--> znaci, poslednju element u deque je sadasnost, ostali su proslost (recording from past frames)
            if len(coordinates_counting[trac_id])!=0: #tj. nije prazan recording from the past for vehicle with trac_id
                if (ymax > line_points[0][1]) and (coordinates_counting[trac_id][0] < line_points[0][1]):
                    contador_in +=1 #-->ovo je brojac za sve tipove vozila, dakle ukupan broj za jedan smer
                    
                    #necessary for adding info per category: 15.7.2024.
                    veiculos_contador_in[det[3]] += 1 #det[3] shows class, i.e. vehicle category (see cells below)
                elif (ymax < line_points[0][1]) and (coordinates_counting[trac_id][0] > line_points[0][1]):
                    contador_out += 1 #-->ovo je brojac za sve tipove vozila, dakle ukupan broj za drugi smer
                    
                    #necessary for adding info per category: 15.7.2024.
                    veiculos_contador_out[det[3]] += 1 #det[3] shows class, i.e. vehicle category (see cells below)
                    
                else:
                    pass


        #drawing the number of vehicles in\out
        cv2.putText(img=annotated_frame, text=f'Left:{contador_in}', 
                    org= (850,300),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 255),thickness=2)
        
        cv2.putText(img=annotated_frame, text=f'Right:{contador_out}', 
                    org= (850,400),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 255),thickness=2)



        
        #*******************************************
        #drawing the number of vehicles in\out
        cv2.putText(img=annotated_frame, text='Vehicles in left lane:', 
                    org= (30,30), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=1, color=(0, 0, 0),thickness=1)
        
        cv2.putText(img=annotated_frame, text='Vehicles in right lane:', 
                    org= (int(1150* scale_percent/100 ),30), 
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 0),thickness=1)

        
        
        contador_in_plt = [f'{myDict[k]}: {i}' for k, i in veiculos_contador_in.items()]
        contador_out_plt = [f'{myDict[k]}: {i}' for k, i in veiculos_contador_out.items()]

        #drawing the counting of type of vehicles in the corners of frame 
        xt = 50
        for txt in range(len(contador_in_plt)):
            xt +=30
            cv2.putText(img=annotated_frame, text=contador_in_plt[txt], 
                        org= (30,xt), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                        fontScale=1, color=(0, 0, 0),thickness=1)
            
            cv2.putText(img=annotated_frame, text=contador_out_plt[txt],
                        org= (int(1150 * scale_percent/100 ),xt), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1, color=(0, 0, 0),thickness=1)

        #*******************************************
        
        
    
        frame_list.append(annotated_frame)

    # Save the video temporarily
    #temp_file_video = NamedTemporaryFile(delete=False, suffix='.mp4') #umesto da mu ja smaisljam ime
    #return frame_list
    #print(len(frame_list))

    return f"Frame {len(frame_list)} added"
    
    #return "Video saved successfully."



    ##closing all open windows 
    #cv2.destroyAllWindows()

# Function to save and show video from frames
def save_and_show_video():
    #print(len(frame_list))
    if len(frame_list) == 0:
        return "No frames to create video", None
    
    video_path = creating_video_from_frames(frame_list)
    return f"Video created with {len(frame_list)} frames", video_path



with gr.Blocks() as demo:
    # Image upload component for frames
    frame_input = gr.Video(label="Upload a video file (.mp4, .avi, etc.)")
    
    # Button to add frames
    add_frame_button = gr.Button("Create Frames from Uploaded Video")
    
    # Video output component
    video_output = gr.Video(label="Created Video")
    
    # Button to save and show video
    save_video_button = gr.Button("Save and Show Video")
    
    # Text output component
    text_output = gr.Text(label="Status")

    # Define Gradio interface layout and interactions
 
    
    add_frame_button.click(main_function_vehicles, inputs=frame_input, outputs=text_output)
    
    save_video_button.click(save_and_show_video, outputs=[text_output, video_output])


# Launch the interface
demo.launch()



