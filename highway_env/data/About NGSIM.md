# About NGSIM
Columns in this Dataset

| Column       | Type       | Description                                                  |
| ------------ | ---------- | ------------------------------------------------------------ |
| Vehicle_ID   | Number     | Vehicle identification number (ascending by time of entry into section). REPEATS ARE NOT ASSOCIATED. |
| Frame_ID     | Number     | Frame Identification number (ascending by start time)        |
| Total_Frames | Number     | Total number of frames in which the vehicle appears in this data set |
| Global_Time  | Number     | Elapsed time in milliseconds since Jan 1, 1970.              |
| Local_X      | Number     | Lateral (X) coordinate of the front center of the vehicle in feet with respect to the left-most edge of the section in the direction of travel. |
| Local_Y      | Number     | Longitudinal (Y) coordinate of the front center of the vehicle in feet with respect to the entry edge of the section in the direction of travel. |
| Global_X     | Number     | X Coordinate of the front center of the vehicle in feet based on CA State Plane III in NAD83. Attribute Domain Val |
| Global_Y     | Number     | Y Coordinate of the front center of the vehicle in feet based on CA State Plane III in NAD83. |
| v_length     | Number     | Length of vehicle in feet                                    |
| v_Width      | Number     | Width of vehicle in feet                                     |
| v_Class      | Number     | Vehicle type: 1 - motorcycle, 2 - auto, 3 - truck            |
| v_Vel        | Number     | Instantaneous velocity of vehicle in feet/second.            |
| v_Acc        | Number     | Instantaneous acceleration of vehicle in feet/second square. |
| Lane_ID      | Number     | Current lane position of vehicle. Lane 1 is farthest left lane; lane 5 is farthest right lane. Lane 6 is the auxiliary lane between Ventura Boulevard on-ramp and the Cahuenga Boulevard off-ramp. Lane 7 is the on-ramp at Ventura Boulevard, and Lane 8 is the off-ramp at Cahuenga Boulevard. |
| O_Zone       | Plain Text | Origin zones of the vehicles, i.e., the place where the vehicles enter the tracking system. There are 11 origins in the study area, numbered from 101 through 111. Please refer to the data analysis report for more detailed information. |
| D_Zone       | Plain Text | Destination zones of the vehicles, i.e., the place where the vehicles exit the tracking system. There are 10 destinations in the study area, numbered from 201 through 211. Origin 102 is a one-way off-ramp; hence there is no associated destination number 202. Please refer to the data analysis report for more detailed information. |
| Int_ID       | Plain Text | Intersection in which the vehicle is traveling. Intersections are numbered from 1 to 4, with intersection 1 at the southernmost, and intersection 4 at the northernmost section of the study area. Value of “0” means that the vehicle was not in the immediate vicinity of an intersection and that the vehicle instead identifies with a section of Lankershim Boulevard (Section_ID, below). Please refer to the data analysis report for more detailed information. |
| Section_ID   | Plain Text | Section in which the vehicle is traveling. Lankershim Blvd is divided into five sections (south of intersection 1; between intersections 1 and 2, 2 and 3, 3 and 4; and north of intersection 4). Value of “0” means that the vehicle does not identify with a section of Lankershim Boulevard and that the vehicle was in the immediate vicinity of an intersection (Int_ID above). Please refer to the data analysis report for more detailed information |
| Direction    | Plain Text | Moving direction of the vehicle. 1 - east-bound (EB), 2 - north-bound (NB), 3 - west-bound (WB), 4 - south-bound (SB). |
| Movement     | Plain Text | Movement of the vehicle. 1 - through (TH), 2 - left-turn (LT), 3 - right-turn (RT). |
| Preceding    | Number     | Vehicle ID of the lead vehicle in the same lane. A value of '0' represents no preceding vehicle - occurs at the end of the study section and off-ramp due to the fact that only complete trajectories were recorded by this data collection effort (vehicles already in the section at the start of the study period were not recorded). |
| Following    | Number | Vehicle ID of the vehicle following the subject vehicle in the same lane. A value of '0' represents no following vehicle - occurs at the beginning of the study section and onramp due to the fact that only complete trajectories were recorded by this data collection effort (vehicle that did not traverse the downstream boundaries of the section by the end of the study period were not recorded). |
| Space_Headway | Number | Space Headway in feet. Spacing provides the distance between the front center of a vehicle to the front-center of the preceding vehicle. |
| Time_Headway | Number | Time Headway in seconds. Time Headway provides the time to travel from the front-center of a vehicle (at the speed of the vehicle) to the front-center of the preceding vehicle. A headway value of 99 |
|Location| Plain Text |Name of street or freeway|

