first `pip install opencv-python` and `pip install ultralytics`.

The models are stored in "weights" directory. The two models are "best.pt" and "last.pt". Currently using "last.pt"
for as it is most refined. 

The model in trained for 30 epochs, with a total of 500 items. The items have various images of multiple or single
cardboard boxes, filtered with noise. The in depth details for model traning are in "results.csv".
![epoch result](results.png?raw=true|400)

Finally, all the testing images are in "images" directory. The default is set to "images/303_jpg.rf.c3390072384e5dbfe04b00b0c37c4892.jpg"
but can be changed by modifying the "img" variable in main.
![epoch result](images/303_jpg.rf.c3390072384e5dbfe04b00b0c37c4892.jpg?raw=true)

Example will be similar to the "output.png" in a new window for the default image.
![epoch result](output.png?raw=true)
