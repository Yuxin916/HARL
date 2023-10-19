from Prison_Escape.environment.abstract_object import DetectionObject
import Prison_Escape.environment.helicopter
import Prison_Escape.environment.search_party


class Fugitive(DetectionObject):
    def __init__(self, terrain, location):
        """
        Fugitive defines the fugitive. Initializes detection parameters.
        :param terrain: a terrain instance
        :param location: a list of length 2. For example, [5, 7]
        """
        DetectionObject.__init__(self, terrain, location, detection_object_type_coefficient=0.0)
        # NOTE: the detection_object_type_coefficient is variant for fugitive as it is detecting different objects

    def detect(self, location_object, object_instance):
        """
        Determine detection of an object based on its location and type of the object
        The fugitive's detection of other parties is different other parties' detection of the fugitive
        The fugitive's detection of other parties depends on what the party is.
            evader对不同类型的追捕者的检测能力不同

            - 当evader检测到helicopter时，detection_object_type_coefficient=0.5
            - 当evader检测到search_party时，detection_object_type_coefficient=0.75
            - 当evader检测到camera时，detection_object_type_coefficient=1.0

        :param location_object:
        :param object_instance: the instance referred to the object the fugitive is detecting.
        :return: [b,x,y] where b is a boolean indicating detection, and
        [x,y] is the location of the object in world coordinates if b=True,
        [x,y]=[-1,-1] if b=False
        """
        if isinstance(object_instance, Prison_Escape.environment.helicopter.Helicopter):
            self.detection_object_type_coefficient = 0.5
            return DetectionObject.detect(self, location_object), DetectionObject.detection_range(self)
        elif isinstance(object_instance, Prison_Escape.environment.search_party.SearchParty):
            self.detection_object_type_coefficient = 0.75
            return DetectionObject.detect(self, location_object), DetectionObject.detection_range(self)
        elif isinstance(object_instance, Prison_Escape.environment.camera.Camera):
            self.detection_object_type_coefficient = 1.0  #TODO： evader为什么没有检测到camera的能力？
            return DetectionObject.detect(self, location_object), DetectionObject.detection_range(self)
        else:
            raise NotImplementedError
