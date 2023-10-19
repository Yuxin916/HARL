from Prison_Escape.environment.abstract_object import DetectionObject


class Camera(DetectionObject):
    def __init__(self, terrain, location, known_to_fugitive, detection_object_type_coefficient):
        """
        Camera defines camera objects. Initializes detection parameters.
        :param terrain: a terrain instance
        :param location: a list of length 2. For example, [5, 7]
        :param known_to_fugitive: boolean denoting whether the camera is known to the fugitive
        """
        assert detection_object_type_coefficient == 1.0

        DetectionObject.__init__(self, terrain, location, detection_object_type_coefficient)
        self.known_to_fugitive = known_to_fugitive

    def detect(self, location_object):
        """
        当camera检测到evader时，detection_object_type_coefficient=1.0
        """
        return DetectionObject.detect(self, location_object), DetectionObject.detection_range(self)