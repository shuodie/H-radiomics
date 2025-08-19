import logging
import os
import SimpleITK as sitk
import pandas as pd
import radiomics
from radiomics import featureextractor

# Get the directory containing the patient data
patient_dir = os.path.abspath(os.path.join('ba yu bin', os.pardir))

if not os.path.exists(patient_dir):
    print(f"Error: Directory '{patient_dir}' does not exist.")
    exit()


# Loop through the patient directories
for patient_name in os.listdir(patient_dir):
    patient_path = os.path.join(patient_dir, patient_name)
    if os.path.isdir(patient_path) and patient_name not in ['.idea', 'exampleSettings', 'H_radiomics','3D reconstuction', 'logistic regression', 'H_radiomics_11new']:
        roi_path = os.path.join(patient_path, 'label_cropped.nrrd')
        cluster_dir = os.path.join(patient_path, 'cluster_labels')

        roi = sitk.ReadImage(roi_path)
        roi_array = sitk.GetArrayFromImage(roi)
        zong_roi = (roi_array == 1).sum()

        # Create an empty DataFrame to store the results for the current patient
        result_df = pd.DataFrame(
            columns=['cluster_name', 'cluster_ratio'])

        # Loop through the cluster label files
        for cluster_name in os.listdir(cluster_dir):
            if cluster_name.endswith('.nrrd'):
                maskName = os.path.join(cluster_dir, cluster_name)
                imageName1 = os.path.join(patient_path, 'T1_cropped.nrrd')
                imageName2 = os.path.join(patient_path, 'T2_cropped.nrrd')

                if imageName1 is None or maskName is None or imageName2 is None:
                    # Something went wrong, in this case PyRadiomics will also log an error
                    print(f'Error getting testcase for {patient_name}/{cluster_name[:-5]}!')
                    continue

                # Read the cluster image
                cluster = sitk.ReadImage(maskName)
                cluster_array = sitk.GetArrayFromImage(cluster)
                cluster_roi = (cluster_array == 1).sum()
                ratio = cluster_roi / zong_roi

                # Set up logging
                logger = radiomics.logger
                logger.setLevel(logging.DEBUG)
                # handler = logging.FileHandler(filename=f'testLog_{patient_name}_{cluster_name[:-5]}.txt', mode='w')
                # formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
                # handler.setFormatter(formatter)
                # logger.addHandler(handler)

                # Initialize feature extractor
                paramsFile = os.path.abspath(r'exampleSettings\exampleMR_3mm.yaml')
                extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)

                # Calculate features
                print(f"Calculating features for T1 in {patient_name}/{cluster_name[:-5]}")
                featureVector1 = extractor.execute(imageName1, maskName)
                print(f"Calculating features for T2 in {patient_name}/{cluster_name[:-5]}")
                featureVector2 = extractor.execute(imageName2, maskName)

                # Add prefix to keys in featureVector1
                prefixedFeatureVector1 = {'T1_' + key: value for key, value in list(featureVector1.items())[37:]}

                # Add prefix to keys in featureVector2
                prefixedFeatureVector2 = {'T2_' + key: value for key, value in list(featureVector2.items())[37:]}

                # Merge the two dictionaries
                mergedFeatureVector = {'cluster_name': cluster_name[:-5], 'cluster_ratio': ratio, **prefixedFeatureVector1, **prefixedFeatureVector2}

                # Add the current cluster's features to the result DataFrame
                result_df = result_df.append(mergedFeatureVector, ignore_index=True)

        # Save the results to an Excel file in the current patient's directory
        result_df.to_excel(os.path.join(patient_path, 'features.xlsx'), index=False)