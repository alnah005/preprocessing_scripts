# preprocessing_scripts

Preprocessing repository for tasks that utilize aggregation_for_caesar Zooniverse tool

## Introduction

If you've stumbled across this repository, I'm sad to tell you that there isn't documentation for every single method/functionality in the scripts. However, I hope the README.md files in each directory will help in deciphering what the heck is going on. Also, I hope that each file/functionality being somewhat small/modular will help. Most of these scripts were generated iteratively and quickly, so it was difficult to have good programming standards/documentations along the process. There are some places where I put print statements that instruct you to do something specific, so make sure you are aware of those. With the current state of this repository, feel free to refactor and improve the structure of the scripts and make them more usable for whoever is after you.

## Structure of this repo

Each directory resembles a project/task that is completely independent (seperable) from other tasks. Each directory has it's own README.md file that describes the specific files in the directory.

## Directories

1. Solarjets: Contains the code generated during the solarjets beta tests.

2. HTR: Contains the code required to preprocess ACLS data to train the line detector and transcriber models.

3. HTR_Gold_standard_processing: Contains the code required to preprocess the gold standard data that Rebecca at this moment in time is transcribing.
