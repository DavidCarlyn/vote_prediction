No special action is required to run this code, it will run as part of the feature extraction process.
This file contains a brief description of the code.  Comments in the code itself provide more detail.

Audio feature processing was done with the Praat software through a Python library called Parselmouth.  
Basic psuedocode is as follows:

for each year folder:
	for each case in the folder:		
		for each audio file:
			
			create sound, pitch, and pulses objects.
			
			call Praat and get the voice report string.
			
			parse the voice report string into separate features.
			
			record features
			
			
Some difficulties that arose included missing, corrupt, and too short files.  To compensate for these shortcomings,
various countermeasures like try/except blocks and size checks were used.  Any such erroneos files were also recorded in separate lists.

parseing the string for the features is necessary as Parselmouth is not a complete library.  It contains direct calls for most of the features, but not all.
Hence, it was determined that simply parsing the complete report string would be the best way to aquire all the necessary data.

The parameters used for the Praat calls are default values that are suitable for regular voice audio as is.
