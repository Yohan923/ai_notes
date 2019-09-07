# Detections

Intuition:

- outputs can be a set of cordinates used as landmarks.
- output is a vector of:
    1. $p$ - probability of oject
    2. $b_x, b_y$ - cordinates of centre of bounding box
    3. $b_h, b_w$ - height and width of the bounding box
    4. $c_1...c_n$ - the object bounded by the box

## Sliding window:

- slide a fixed size window with some stride to crop a small section of the image into a conv net. 
- with each pass the window gets bigger
- large computation cost, especially with conv nets
- coarse find can lower accuracy

Use conv nets to implement sliding windows:

- max pool layer of 2 represents a stride of 2 in the input image
- by doing the following we can do sliding window on the entire image by sharing the feature that are common to each of the crops.

![**Sliding Window**](images/Sliding_window.jfif)

## Yolo algorithm(You only look once)

- divide image into grids and applie detection algorithm to each of the grids.
- take midpoint of the object and assign each object to the grid cell containing the center point.
- cordinates are relative to the grid cell

With Anchors and Non-max:
- for each grid cell get predicted boxes with anchor boxes
- supress low probability boxes
- use non-max supression for each of the object class independently.

## Intersection over union(IoU):

measure of overlaping between bounding boxes.

$$
    IoU = \frac{intersection \ area}{union \  area}
$$

- in general IoU >= 0.5 is ok

## Non-max Supression:

make sure your algorithm only detect an object once.

Algorithm:

1. select a box with highest probability of a found object
2. supress other boxes that has a high IoU with the box
3. repeat 1. and 2. for the remaining boxes.

## Anchor boxes:

detect multiple objects in a grid cell

- each object in the image is now assigned to the grid cell that has its mid point and the anchor box that has the highest IoU
- doesn't handel objects with same anchor box shape
- doesn't handel more objects than anchor boxes

![**Anchor boxes**](images/Anchor.jfif)

## Region Proposal(R-CNN):

- only run CNN on regions that makes sense ie. regions that may have objects.
- region segmentation algorithm, find some blobs to run CNN on, save computation
- this is still quite slow though
- faster R-CNN use CNN to propose regions.
- but still slow.

