import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython import display
from IPython.display import HTML
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def crop(img, xmin=700,xmax=4300,vmin=2500,vmax=16000):
    '''Remove matplotlib canvas from image.
    Returns: cropped image'''
    return img[xmin:xmax,vmin:vmax]


def channel_last(img):
    '''Convert 'channel_first' to 'channel_last' format.
    
    '''
    return np.rollaxis(img, 0, 3)  


def play_raw_data(data_frame, imgs, f_init=0, f_end=300):
    """Plat sucession of frames from input data. Annotate CAN readings.
    
    data: CAN_dataframe
    f_init: initial frame
    f_end: end frame
    """
    fig, ax = plt.subplots(1)
    for i in range(f_init,f_end,3):
        image_id = data_frame['cam_index'].iloc[i]
        img = imgs[int(image_id)]
        img = annotate(channel_last(img), data_frame.iloc[i], opacity=0.6)
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.imshow(img)
        plt.axis('off')
    plt.close()
    
    
def simulator(model,data_frame,imgs,f_init=230,f_end=300):
    """Plot images and autopilot model prediction.
    
    model: Pre-trained Keras model
    data: CAN_dataframe
    f_init: initial frame
    f_end: end frame
    """
    fig, ax = plt.subplots(1)
    plt.title('Model predictions')
    plt.axis('off')
    total_frames = []
    for i in range(f_init,f_end,1):
        image_id = data_frame['cam_index'].iloc[i]
        img = imgs[int(image_id)]
        img = channel_last(img)
        angle_rad = float(model.predict(np.expand_dims(img, axis=0)))
        angle = angle_rad
        real_angle = data_frame['steering_angle'].iloc[i]
        error = angle - (real_angle)

        if np.abs(error) < 6:   # Fill green if prediction is accurate
            left = 130 
            rect = patches.Rectangle((left, 85),40,20, angle=angle,
                                     linewidth=1,edgecolor='lightgreen',facecolor='lightgreen',
                                     label='{:2f} degrees. Error {:2f}'.format(angle,error),
                                     alpha=0.15
                                     )
            rect2 = patches.Rectangle((left, 85),40,20, angle=real_angle,
                                     linewidth=2,edgecolor='blue',facecolor='none')
            patch = ax.add_patch(rect)
            patch2 = ax.add_patch(rect2)

        else:  # Fill red if prediction is far from target

            left = 130
            rect = patches.Rectangle((left, 85),40,20, angle=angle,
                                     linewidth=1,edgecolor='lightgreen',facecolor='red',
                                     label='{:2f} degrees. Error {:2f}'.format(angle,error),
                                     alpha=0.10
                                     )
            rect2 = patches.Rectangle((left, 85),40,20, angle=real_angle,
                                     linewidth=2,edgecolor='blue',facecolor='none')#,
            patch = ax.add_patch(rect)
            patch2 = ax.add_patch(rect2)
            
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.imshow(img)
        plt.legend()
        plt.axis('off')
        # Grab the pixel buffer and dump it into a numpy array
        fig.canvas.draw()
        X = np.array(fig.canvas.renderer._renderer)
        X = crop(X,60,225,55,388)
        # Append frames to render video
        total_frames.append(X)
        patch.remove()
        patch2.remove()

    plt.close()

    return total_frames
    
    
def annotated_plot(data_frame, imgs):
    """Plot single frame with CAN annotations.
    
    data: CAN_dataframe
    """
    image_id = data_frame['cam_index'].iloc[i]
    #img = imgs[int(image_id)]
    img = annotate(channel_last(img), data_frame.iloc[i], opacity=0.6)
    plt.imshow(img)
    plt.axis('off')
    
    
def annotate(img, dataframe_row, opacity=0.6):
    '''Annotate image with CAN readings.

    '''
    meta = dataframe_row['steering_angle speed accel brake'.split()]
    
    frame_can = 'st_angle={} speed={}MPH accel={}G brake={}'.format(meta[0].round(2), 
                                                                      meta[1].round(2), 
                                                                      meta[2].round(2),
                                                                      meta[3].round(2))
    frame = img.copy()
    overlay = frame.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    y0, dy = 15, 10
    for i, line in enumerate(frame_can.split()):
        y = y0 + i*dy
        cv2.putText(overlay, line, (10, y), font, 0.3, (255, 255, 255), 1)

    #cv2.circle(overlay, (133, 132), 12, (0, 255, 0), -1)
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
    
    return frame

