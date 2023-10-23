# %%
from ultralytics import YOLO
import cv2

def main()->None:
    img = "images/303_jpg.rf.c3390072384e5dbfe04b00b0c37c4892.jpg"

    # model = YOLO("yolov8s.pt")
    # model.train(data="ignore/data.yaml", epochs=30, batch=7)
    # after being trained:
    model = YOLO("weights/last.pt")
    #The weights folder has 2 files, best.pt, last.pt, the epoch with best returns and last epoch respectively.

    results = model(img)

    img = cv2.imread(img, 1)

    for box in results[0].boxes:
        cords = box.xyxy[0].tolist()
        #cords are in format: [209, 145, 434, 324], with first two being top left corner, lsat 2 being bottom right
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)
        print("Coordinates:", cords)
        print("Probability:", conf)

        cv2.putText(img, "conf"+str(conf), [cords[0], cords[1]-5], 2, fontScale=0.5, color= (0,0,0))
        img = cv2.rectangle(img, cords[:2], cords[2:], (0,0,0), 2)


    if img is not None:
        # Display the image in a window
        cv2.imshow('Image', img)

        # Wait for a key press indefinitely, and close the window when a key is pressed
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to load the image.")

        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
