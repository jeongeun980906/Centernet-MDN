cd /data/opensets/voc

# mkdir /data/private/voc
# mkdir /data/private/voc/images
# mkdir /data/private/voc/annotations

# cp VOCdevkit/VOC2007/JPEGImages/* /data/private/voc/images/
# cp VOCdevkit/VOC2012/JPEGImages/* /data/private/voc/images/
wget https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip

mkdir /data/private/voc
unzip PASCAL_VOC.zip
rm PASCAL_VOC.zip
mv PASCAL_VOC /data/private/voc/annotations/
cd /data/private/Centernet-MDN/utils
python merge_pascal_json.py