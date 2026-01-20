### Motion Ratio Calculation - CRITICAL IMPLEMENTATION DETAILS

**Source:** von Lühmann et al. (2020) - Front. Neurosci. 14:579353

The motion ratio MUST be calculated on **optical density (OD)**, not raw intensity. The MATLAB code from the quality analysis script confirms this:

```matlab
currentOD = hmrR_Intensity2OD(currentfile.data);
[tInc, tIncCh] = hmrR_MotionArtifactByChannel(currentOD, currentfile.probe, [], [], [], 0.5, 0.5, 20, 5);
ChannelMR = 1 - sum(tIncCh{1}) ./ length(tIncCh{1});
```

**hmrR_MotionArtifactByChannel Parameters:**
1. **tMotion = 0.5:** Time threshold for motion detection window
2. **tMask = 0.5:** Time to mask around detected motion
3. **STDEVthresh = 20:** Standard deviation multiplier threshold
4. **AMPthresh = 5:** Amplitude threshold in OD units

**Algorithm (from Homer3 documentation):**
1. Convert intensity to optical density first
2. Calculate temporal derivative of OD signal
3. Detect motion using two methods:
   - **Peak-to-peak amplitude:** Flag where local change exceeds AMPthresh (5 OD units)
   - **Standard deviation:** Flag where local std exceeds STDEVthresh × global std (20×)
4. Combine flagged segments
5. Calculate ratio: flagged_time / total_time

**Why OD not Intensity?**
- OD is logarithmically related to intensity: OD = -log10(I/I₀)
- Motion causes multiplicative changes in intensity but additive changes in OD
- OD units make thresholds more consistent across different baseline intensities
