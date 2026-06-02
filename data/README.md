# Data Directory

The backend writes generated ChEMBL activity cache and processed molecule records here:

- `raw_chembl_activities.json`
- `processed_molecules.json`

Those generated files are ignored so the repository stays lightweight. Recreate them with `POST /index/build` or the README setup command.

