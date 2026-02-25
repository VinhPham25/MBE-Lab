# Surgical Pattern Cutting fNIRS Dataset

## Dataset Description:
fNIRS Recordings saved in .Snirf file format.
+ Expert/Novice-15 recordings across 14 subjects (69 trials)
+ Learning Curve-120 recordings across 13 subjects (760 trials)
+ Short-Term tES-60 recordings across 12 subjcts (312 trials)
+ Long-Term tES-274 recordings across 21 subjects (2181 trials)

## Experimental Description:
The PC task is one of the five psychomotor tasks in the FLS program, where the goal is to use laparoscopic tools to cut a marked piece of gauze as quickly and as accurately as possible. The subjects must operate in a volume-constrained 3D environment while looking at the 2D monitor screen for visual feedback. All participants were instructed on how to perform the task with standardized verbal instructions (or video) indicating the goal of the task and rules for task completion. All the experiments were conducted in adequate lighting. 

### Subject Demographics
|Dataset|Expert/Novice| Learning Curve| Short-Term tES| Long-Term tES|
|:--------:|:--------:|:----------:|:---------:|:----------:|
|Sample Size| 14|13|12|21|
|Mean Age |25|26|25|23|
|Sex (M/F)| 12/7|8/3|1/12|15/6|

## Data Collection Hardware
#### Expert/Novice and Learning Curve Studies
Number of Long Separation Channels: 25
Number of Short Separation Channels: 8 (7 in Learning Curve dataset due to hardware issues)
Sampling Rate: 20 Hz
Optode Coverage: Prefrontal Cortex, Primary Motor Cortex, Supplementary Motor Area
Wavelengths: 690/830
Acquisition System: TechEn CW6

#### Short/Long-term tES Studies
Number of Long Separation Channels: 28
Number of Short Separation Channels: 8
Sampling Rate: 7.8125 Hz
Optode Coverage: Prefrontal Cortex, Primary Motor Cortex, Supplementary Motor Area
Wavelengths: 760/850
Acquisition System: NIRx NIRScout

### Data Usage Notes:
The fNIRS recordings are in Snirf file format and can be opened and processed using the Homer3 Matlab Toolbox or the MNE-NIRS Python toolbox. Stimulus timestamps marked as condition 1 *(Vertical blue lines in Homer3)* correspond to the beginning of each surgical pattern cutting trial, while those marked as condition 2 *(Vertical red lines in Homer3)* correspond to the end of each trial. Each folder contains all of the recordings for each subject, with each recording containing all of the trials of a subject during that day. 
In the expert/novice dataset, we use A to denote attending surgeons and R to denote residents in the file names
In the learning curve and long-term tES we use C/S and L respectively to denote subject numbers, and the number following the subject number on each file denotes the session/training day
Unlike the expert/novice dataset which only consists of one recording for each subject and the learning curve and long-term tES datasets where each recording corresponds to each day of training, the short-term tES study recorded multiple sessions in a single day for each participant, and as such, file nomenclature slightly differs compared to the others. Below we provide a key to the file nomenclature:
+ 0-PT- PreTrials before starting experiment
+ 1-BS1- Trials Before Stimulation of first testing condition (1st recording after pretest)
+ 3-BS2- Trials Before Stimulation of 2nd testing condition (3rd recording after pretest)
+ 2/4-T- Trials after tRNS stimulation (either the 2nd or 4th recording after pretest due to counterbalanced design)
+ 2/4-S- Trials after Sham stimulation (either the 2nd or 4th recording after pretest due to counterbalanced design)

Below we provide additional notes regarding individual recordings that may be relevant to the use of this data:
|Subject|Recording| Note|
|:--------:|:--------:|:--------:|
||**Expert/Novice**|
|R2| FLS| Beginning of Trial 5 estimated|
|R3| FLS| End of Trial 3 Estimated|
|R9| FLS| Split into 2 recordings with Trial 3 missing|
||**Learning Curve**|
|FLS2| Transfer| Beginning of Excision estimated|
|FLS4| Transfer| Beginning of Excision estimated|
|FLS5| Session 5| Beginning of Trial 5 estimated|
|FLS5|  Session 7| Longer break than usual between 2 trials|
|FLS5| Retention| File missing stimulus marks due to hardware issues|
|FLS5| Transfer| Missing duration/score|
|FLS6| Session 1| Beginning of Trial 1 estimated|
|FLS6| Session 7| Beginning of Trial 1 estimated|
|FLS7| Session 1| Beginning of Trial 1 estimated|
|Ctrl1 | FLS|fNIRS recording missing|
|Ctrl2 | Final| Did not complete post-test or Transfer|
|Ctrl4 | FLS |Beginning of Trial 1 estimated|
||**Short-term tES**|
|S9 | Pretrial 1| Trial 1 missing from fNIRS recording|
|S6| Training 1| Session split into 2 recordings with Trial 3 missing|
||**Long-term tES**|
|L5| Session 5| Session split into 2 recordings with the end of Trial 2 missing|
|L6| Session 8| Subject performed an extra trial|
|L24|Session 8| End of Trial 3 estimated|