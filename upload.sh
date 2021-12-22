rm -r ./dist
rm -r ./src/torchspider.egg-info
python3 -m build
python3 -m pip install --upgrade twine 
python3 -m twine upload dist/*