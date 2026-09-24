# Atrial Regionalisation

Interactively place the named landmarks of the EHRA/EACVI bi-atrial
segmentation model (Althoff et al., Europace 2025;27:euaf134) on a left or
right atrial surface mesh, then compute the resulting 15-segment
cell-based regionalisation.

## Prerequisites

Refer to the `atrial_regionalisation` [GitHub page](https://github.com/arnovonkietzell/atrial_regionalisation) for general instructions for this WIP. Make sure to follow the steps on how to set the interpreter and root directory.

## Running this WIP

- The input to this WIP (`case_1`) should be a clipped left or right atrial
  surface mesh - the mitral/tricuspid valve and veins should already be clipped open as boundary holes.
- Set the `chamber` widget to `LA` or `RA`. 
- Running the WIP opens a window: right-click on the mesh to place the
  landmark named in the panel on the right (a reference image cropped from
  the paper tracks the current landmark automatically). Some landmarks are
  a closed loop or open path of several points - click each point in turn,
  then press "Finish loop/path" once enough are placed. Use "Undo" to step
  back one point/landmark at a time, or "Reset all landmarks" to start
  over.
- Once every landmark is placed, press "Compute regions" to preview the
  15-segment labelling.
- Press "Confirm & Close" to finish. A new case is created with a
  `cell_region` field (values 1-15, 0 = unmatched/unlabelled) holding the
  computed segmentation.
- Closing the window any other way (e.g. the title bar's close button)
  discards the run - no output case is created.
