"""The data structure module holds model classes."""
import SimpleITK as sitk


class BrainImage:
    """Represents a brain image."""
    
    def __init__(self):
        """Initializes a new instance of the BrainImage class."""
        self.t1 = None
        self.t2 = None
        self.ground_truth = None
        
