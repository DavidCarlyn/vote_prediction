import parselmouth
import numpy as np
import glob
import os

def str_to_float(str_val):
    if str_val == '--undefined--':
        return -1.0
    else:
        return float(str_val)

# Choose which year to run this for
year = "2015"
yearFolderPath = os.path.join("D:", "Datasets", "SC_Audio", year)

# Output file info
outputFile = year + ".txt"
np.set_printoptions(linewidth=np.inf)

# Get the list of all the case folder paths.  Have to do this as the case numbers aren't contiguous.
# Also counting the number of case folders.
casePaths = []
for subdir, dirs, _ in os.walk(yearFolderPath):
    for dir in dirs:
        casePaths.append(os.path.join(subdir, dir))

# Loop over each case in the specified year
for casePath in casePaths:

    # Count case files in folder and preallocate a NumPy vector.  Doing this to avoid returning new arrays with .append().
    # Also declare a list for the feature values
    num_files = len([f for f in os.listdir(casePath) if os.path.isfile(os.path.join(casePath, f))])
    num_features = num_files * 26
    feature_vector = np.zeros(num_features, dtype='float32')

    fileIndex = 0

    # test code
    fileCount = 0
    print (casePath)

    # Loop over all the mp3 files in the case folder
    for mp3 in glob.glob(casePath + "/*.mp3"):

        # These are apparently the 3 objects you need
        sound = parselmouth.Sound(mp3)
        pitch = sound.to_pitch()
        pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")

        # get voice report and create a list for the features
        voice_report_str = parselmouth.praat.call([sound, pitch, pulses], "Voice report", 0.0, 0.0, 75, 600, 1.3, 1.6, 0.03, 0.45)
        VPRsplit = voice_report_str.split()
        feature_list = []
    
        # append all the parsed features into the list.  I counted the strings manually assuming the report would always have the same format.
        feature_list.append(str_to_float(VPRsplit[11]))
        feature_list.append(str_to_float(VPRsplit[15]))
        feature_list.append(str_to_float(VPRsplit[19]))
        feature_list.append(str_to_float(VPRsplit[23]))
        feature_list.append(str_to_float(VPRsplit[27]))
    
        feature_list.append(str_to_float(VPRsplit[33]))
        feature_list.append(str_to_float(VPRsplit[37]))
        feature_list.append(str_to_float(VPRsplit[40]))
        feature_list.append(str_to_float(VPRsplit[46]))
    
        feature_list.append(str_to_float(VPRsplit[54].strip('%')))
        feature_list.append(str_to_float(VPRsplit[62]))
        feature_list.append(str_to_float(VPRsplit[67].strip('%')))
    
        feature_list.append(str_to_float(VPRsplit[76].strip('%')))
        feature_list.append(str_to_float(VPRsplit[80]))
        feature_list.append(str_to_float(VPRsplit[84].strip('%')))
        feature_list.append(str_to_float(VPRsplit[87].strip('%')))
        feature_list.append(str_to_float(VPRsplit[90].strip('%')))
    
        feature_list.append(str_to_float(VPRsplit[94].strip('%')))
        feature_list.append(str_to_float(VPRsplit[98]))
        feature_list.append(str_to_float(VPRsplit[102].strip('%')))
        feature_list.append(str_to_float(VPRsplit[105].strip('%')))
        feature_list.append(str_to_float(VPRsplit[108].strip('%')))
        feature_list.append(str_to_float(VPRsplit[111].strip('%')))
    
        feature_list.append(str_to_float(VPRsplit[120]))
        feature_list.append(str_to_float(VPRsplit[124]))
        feature_list.append(str_to_float(VPRsplit[128]))

        # Copy the feature list into the correct cells of the numpy array
        for i in range(26):
            feature_vector[i + fileIndex] = feature_list[i]

        # up the index by 26 to find the next section of the array to fill
        fileIndex = fileIndex + 26
        
        # this is test code
        print ("file " + str(mp3) + " completed!")
        print (fileCount)
        fileCount = fileCount + 1

    # finally write the array as one line in the output file.
    with open(outputFile, 'ab') as f:
        np.savetxt(f, [feature_vector], delimiter=" ")