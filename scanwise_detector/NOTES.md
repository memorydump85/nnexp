Scanwise LiDAR object detector
==============================


Hypotheses
----------
* It is possible to detect objects from single LiDAR scans
* Detections from multiple scans can be combined to improve accuracy


References
----------

### Hough forests
* An Introduction to Random Forests for Multi-class Object Detection, _J.Gall et. al._


Thoughts and Progress
---------------------

* Learn a NN that maps from scan-segment => center-offset
* Use NN to vote for object centers