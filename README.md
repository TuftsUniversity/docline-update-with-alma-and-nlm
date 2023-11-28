# docline-update-with-alma-and-nlm

## Author
- Henry Steele TTS Library Technology Service at Tufts University
- 2023

## Reason and Objective
Tufts like many places with medical libraries faced the challenge of updating their journal holdings in Docline, per instructions in the [Docline User Manual](https://www.nlm.nih.gov/docline/docline_manual/DOCLINE_manual.pdf).   However these instructions seem to assume librarians are going through their holdings one by one to add or update journals from their local collection that conform to Docline ingest standards, and fall within the range of journals that Docline considers to be medical or medical-adjacent

Although it used to be possible to sync Docline holdings with OCLC this appears not be possible anymore as seen [here](https://info.opal-libraries.org/c.php?g=489325&p=8492743)

This unfortunately leaves institutions such as Tufts struggling to determine how we are going to keep our holdings up to date on any kind of regular basis.  To address this problem and to keep participating in Docline and updating our holdings with some regular process, the author from Tufts University wrote this process with these scripts to automate as much as possible the process of finding matches in your local Alma holdings against the total list of NLM journals, which roughly corresponds to the total number of records available in Docline (155,000 in Docline at endpoints such as (https://docline.gov/docline/journals/id/51202/) to 200,000 in [NLM](https://catalog.nlm.nih.gov/discovery/search?vid=01NLM_INST:01NLM_INST) )

## Analysis
In order to find out what methods would work, I needed to compare the three data sets that are needed for this process:
- local docline holdings for our institution, exportable from Docline
- the total list of XML serials.  Choose the latest export from this  [serial XML file release page](https://ftp.nlm.nih.gov/projects/serfilelease/)
- export from Alma Analtyics.  I have put the report I use to get all our serials data from e inventory at /shared/Community/Reports/Institutions/Tufts University/Journals for NLM-Docline

To find the broadest number of matches possible, the script compares your local journal holdings against NLM data by-
- OCLC Number`
- ISSN

to get as many matches based on identifiers as possible.  It then adds the NLM Unique ID to your Alma data, and uses this to match against your Docline holdings, (ignoring the "NLM_" prefix)

##Process

install the Python libraries  with the requirements file:
- python pip install -r requirements.txt

You'll also need to specify local data that at Tufts was determined to be fixed in the Docline data in the secrets_local/secrets_local.py
you can run the script:
- python docline.py
There are two prompts:  1 lets you determine whether you want the script to downline the latest serials file from NLM, and
what folder to put this file in if you have a local copy

and whether you want to compare/update: print, electronic, or both

Your local Docline holdings should be in the
## Output

The process finally outputs 6 reports that you can review and decide how to upload them:

- deleted from ALma in docline
  - delete
- in Alma not in DOcline
  - add
- in Docline only
  - you may want to keep these or at least examine them
they can be left out of ingest because they'll just stay
full match between alma and docline on NLM Unique ID, all holdings fields, and all ranges
there are none of these now
these could be ignored
match on NLM Unique ID and hodlings format, but different range or other holding data
there are two sets here
the Docline data
delete
the alma data
add
