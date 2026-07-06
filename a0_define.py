# 缓存格式，变更后需手动删除缓存文件
major_tags = (
    'ImageType', 'Modality',
    'PatientID', 'StudyInstanceUID', 'SeriesInstanceUID',
    'Manufacturer', 'ManufacturerModelName',
)

minor_tags = (
    'StudyDate', 'StudyTime',
    'PatientName', 'PatientSex', 'PatientAge',
    'NumberOfFrames',
    'StudyDescription', 'SeriesDescription',
    'WindowCenter', 'WindowWidth',
    'ImageOrientationPatient',
)

pair_modalities = ('CT', 'PX')
