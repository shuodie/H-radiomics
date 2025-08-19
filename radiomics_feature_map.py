#!/usr/bin/env python

from __future__ import print_function

import logging
import os

import SimpleITK as sitk
import six

import radiomics
from radiomics import featureextractor, getFeatureClasses, imageoperations
import gc


def tqdmProgressbar():
    """
    This function will setup the progress bar exposed by the 'tqdm' package.
    Progress reporting is only used in PyRadiomics for the calculation of GLCM and GLSZM in full python mode, therefore
    enable GLCM and full-python mode to show the progress bar functionality

    N.B. This function will only work if the 'click' package is installed (not included in the PyRadiomics requirements)
    """
    global extractor

    radiomics.setVerbosity(logging.INFO)  # Verbosity must be at least INFO to enable progress bar

    import tqdm
    radiomics.progressReporter = tqdm.tqdm

def clickProgressbar():
    """
    This function will setup the progress bar exposed by the 'click' package.
    Progress reporting is only used in PyRadiomics for the calculation of GLCM and GLSZM in full python mode, therefore
    enable GLCM and full-python mode to show the progress bar functionality.

    Because the signature used to instantiate a click progress bar is different from what PyRadiomics expects, we need to
    write a simple wrapper class to enable use of a click progress bar. In this case we only need to change the 'desc'
    keyword argument to a 'label' keyword argument.

    N.B. This function will only work if the 'click' package is installed (not included in the PyRadiomics requirements)
    """
    global extractor

    # Enable the GLCM class to show the progress bar
    extractor.enableFeatureClassByName('glcm')

    radiomics.setVerbosity(logging.INFO)  # Verbosity must be at least INFO to enable progress bar

    import click

    class progressWrapper:
        def __init__(self, iterable, desc=''):
            # For a click progressbar, the description must be provided in the 'label' keyword argument.
            self.bar = click.progressbar(iterable, label=desc)

        def __iter__(self):
            return self.bar.__iter__()  # Redirect to the __iter__ function of the click progressbar

        def __enter__(self):
            return self.bar.__enter__()  # Redirect to the __enter__ function of the click progressbar

        def __exit__(self, exc_type, exc_value, tb):
            return self.bar.__exit__(exc_type, exc_value, tb)  # Redirect to the __exit__ function of the click progressbar

    radiomics.progressReporter = progressWrapper

# Get the location of the example settings file
paramsFile = os.path.abspath(r'exampleSettings\exampleVoxel.yaml')

# Get the PyRadiomics logger (default log-level = INFO
logger = radiomics.logger
logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file

# Write out all log entries to a file
# handler = logging.FileHandler(filename='testLog.txt', mode='w')
# formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
# handler.setFormatter(formatter)
# logger.addHandler(handler)

# Initialize feature extractor using the settings file
extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
featureClasses = getFeatureClasses()

# Uncomment one of these functions to show how PyRadiomics can use the 'tqdm' or 'click' package to report progress when
# running in full python mode. Assumes the respective package is installed (not included in the requirements)

tqdmProgressbar()
# clickProgressbar()

print("Active features:")
for cls, features in six.iteritems(extractor.enabledFeatures):
    if features is None or len(features) == 0:
        features = [f for f, deprecated in six.iteritems(featureClasses[cls].getFeatureNames()) if not deprecated]
    for f in features:
        print(f)
        print(getattr(featureClasses[cls], 'get%sFeatureValue' % f).__doc__)

# Get the directory containing the patient data
patient_dir = os.path.abspath(os.path.join('bao xin yue', os.pardir))
for patient_name in os.listdir(patient_dir):
    patient_path = os.path.join(patient_dir, patient_name)
    if os.path.isdir(patient_path) and patient_name not in ['.idea', 'exampleSettings']:
        image_files = [os.path.join(patient_path, 'T1.nrrd'),
                      os.path.join(patient_path, 'T2.nrrd')]
        mask_file = os.path.join(patient_path, 'label.nrrd')
        output_folder = os.path.join(patient_path, 'feature_map')
        os.makedirs(output_folder, exist_ok=True)
        print(f"Calculating features for case: {patient_name}")
        gc.collect()


        for image_file in image_files:
            prefix = os.path.splitext(os.path.basename(image_file))[0]
            image = sitk.ReadImage(image_file)
            mask = sitk.ReadImage(mask_file)
            # 获取预处理后的图像和掩码
            settings = {}
            settings['binWidth'] = 25  # CT用的25，PET用的0.4
            settings['resampledPixelSpacing'] = [1, 1, 1]
            settings['padDistance'] = 0
            settings['interpolator'] = 'sitkBSpline'
            settings['label'] = 1
            # settings['correctMask'] = True
            interpolator = settings.get('interpolator')
            preprocessed_image, preprocessed_mask = extractor.loadImage(image, mask, **settings)

            featureVector = extractor.execute(image_file, mask_file, voxelBased=True)
            bounding_box = featureVector.get('diagnostics_Mask-original_BoundingBox')

            # 保存裁剪后的掩码
            cropped_image_file = os.path.join(patient_path, f"{prefix}_cropped.nrrd")
            cropped_mask_file = os.path.join(patient_path, f"label_cropped.nrrd")
            sitk.WriteImage(preprocessed_image, cropped_image_file, True)
            sitk.WriteImage(preprocessed_mask, cropped_mask_file, True)
            print(f"Stored cropped {prefix} mask in {cropped_mask_file}")
            for key, val in featureVector.items():
                if isinstance(val, sitk.Image):
                    output_file = os.path.join(output_folder, f"{prefix}_{key}.nrrd")
                    sitk.WriteImage(val, output_file, True)
                    print(f"Stored feature {key} in {output_file}")
                else:  # Diagnostic information
                    print(f"\t{key}: {val}")