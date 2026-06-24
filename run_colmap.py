import os
import numpy as np
import sys
import sqlite3
import argparse

IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (model, width, height, array_to_blob(params),camera_id))
        return cursor.lastrowid

def round_python3(number):
    rounded = round(number)
    if abs(number - rounded) == 0.5:
        return 2.0 * round(number / 2.0)
    return rounded

# n_views表示The number of training views.
def pipeline(scene, base_path, n_views, flag):
    llffhold = 8
    # 如果是增强后跑此脚本，就是用12_views_aug
    if flag:
        view_path = str(n_views) + '_views_aug'
    else:
        view_path = str(n_views) + '_views'     # 12_views
    os.chdir(os.path.join(base_path, scene))    # chdir = cd
    os.system('rm -r ' + view_path)
    os.mkdir(view_path)
    os.chdir(view_path)
    os.mkdir('created')
    os.mkdir('sparse')
    os.mkdir('images')
    # 将原本文件夹下的sparse/0中的 .bin -> .txt
    os.system('colmap model_converter  --input_path ../sparse/0/ --output_path ../sparse/0/  --output_type TXT')

    # ================ 读 images.txt，把每张图像的位姿/相机信息按文件名存到 images 字典里 ================
    # images.txt 里每张图片通常占 两行，第一行：图像 id，四元数旋转 qvec，平移 tvec，相机 id，图像文件名
    # 第二行：这张图上所有 2D 特征点及其关联 3D 点
    images = {}
    with open('../sparse/0/images.txt', "r") as fid:
        while True:
            line = fid.readline()
            if not line:    # 如果 readline() 返回空字符串 ""，长度是 0，说明读完了，跳出循环
                break
            line = line.strip() # 去掉这一行首尾的空白字符，"1 0.9 ... image001.png\n"变成"1 0.9 ... image001.png"
            # 跳过：空行，以 # 开头的注释行
            if len(line) > 0 and line[0] != "#":
                # 按空白符把这一行拆成列表，比如一行是：1 0.99 IMG_0001.JPG，
                # 那么 elems 会变成 ["1" "0.99" "IMG_0001.JPG"]
                elems = line.split()
                # elems[0] = IMAGE_ID
                # elems[1:5] = QW,QX,QY,QZ
                # elems[5:8] = TX,TY,TZ
                # elems[8] = CAMERA_ID
                # elems[9] = NAME
                image_name = elems[9]
                fid.readline().split()  # 把当前图像后面的“第二行特征点信息”读掉，但并不使用
                images[image_name] = elems[1:]  # 保存：images["IMG_0001.JPG"] = [QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME]

    # 拿出所有图片名，按元素本身排序，放进 img_list列表，结果是：["000001.png", "000002.png", "000003.png"]
    # sorted(..., key=...) 的意思是：排序前，先对每个元素应用 key 函数，再按函数结果排序。这里 lambda 表示匿名函数，等同于 def f(x): return x
    img_list = sorted(images.keys(), key=lambda x: x)
    # enumerate 会给列表里的每个元素编号，遍历 img_list 中每张图，如果它的下标 idx 不是 llffhold 的倍数，就保留下来。idx 表示下标，c 表示元素本身
    train_img_list = [c for idx, c in enumerate(img_list) if idx % llffhold != 0]
    if n_views > 0:     # n_views 为 3 or 12
        if n_views >= len(train_img_list):
            idx_sub = list(range(len(train_img_list)))
        else:
            # np.linspace(a, b, n) 指：在区间 [a, b] 上均匀取 n 个点。这里相当于在训练集中均匀相隔，共取12张图片，并用 round_python3 四舍五入
            # idx_sub 就是 要保留的训练图像 train_img_list 的下标集合
            idx_sub = sorted(set(round_python3(i) for i in np.linspace(0, len(train_img_list)-1, n_views)))
        train_img_list = [c for idx, c in enumerate(train_img_list) if idx in idx_sub]

    if flag:
        # 把 garden/augimages/ 里的所有图复制到 garden/12_views_aug/images/
        os.system('cp ../augimages/*  images/')
    else:
        for img_name in train_img_list:
            os.system('cp ../images/' + img_name + '  images/' + img_name)
    
    # 把原 sparse 里的 cameras.txt 复制到12_views/created/cameras.txt
    os.system('cp ../sparse/0/cameras.txt created/.')
    # 创建空文件 created/points3D.txt ，之后给定相机参数和图像，重新让 COLMAP 在 few-shot 图像上三角化出新的点云
    with open('created/points3D.txt', "w") as fid:
        pass

    # ============= 这里是对12_views/images进行特征提取和匹配，结果都写进 .db 中 =============
    # 使用更接近旧版 COLMAP 3.8 的接口参数
    # 这里已经在 12_views/ 下了，所以 images指的是 12_views/images里面的
    res = os.popen('colmap feature_extractor --database_path database.db --image_path images --ImageReader.camera_model PINHOLE --SiftExtraction.max_image_size 4032 --SiftExtraction.max_num_features 32768 --SiftExtraction.estimate_affine_shape 0 --SiftExtraction.domain_size_pooling 0').read()
    os.system('colmap exhaustive_matcher --database_path database.db --SiftMatching.guided_matching 1 --SiftMatching.max_num_matches 32768')
    
    # ====== 读取刚刚生成的 COLMAP 数据库 database.db 中的 images 表，取出数据库里记录的图像名顺序 ======
    db = COLMAPDatabase.connect('database.db')  # 连接当前目录下的 SQLite 数据库文件 database.db，返回一个数据库连接对象 db
    db_images = db.execute("SELECT * FROM images")  # 从数据库的 images 表里，把所有记录查出来
    # 遍历查询结果里的每一行 db_image，取第 2 列 db_image[1]，组成一个列表
    # 把数据库中的所有图像名按数据库返回顺序取出来，例如得到：
    # img_rank = ["IMG_0003.JPG", "IMG_0012.JPG", "IMG_0007.JPG", ...]
    img_rank = [db_image[1] for db_image in db_images]  
    print(img_rank, res)
    with open('created/images.txt', "w") as fid:
        for idx, img_name in enumerate(img_rank):
            print(img_name)
            img_name = os.path.basename(img_name)   # 只保留文件名，不保留路径："images/IMG_0005.JPG" -> "IMG_0005.JPG"
            # 所以 flag=True 的本质：增强图沿用原图位姿，但名字换成增强图文件名
            if flag:
                aug_name = img_name.replace('_aug', '') if '_aug' in img_name else img_name
                data = [str(1 + idx)] + [' ' + item for item in images[aug_name]] + ['\n\n']
                if '_aug' in img_name:
                    data = [line.replace(aug_name, img_name) for line in data]
                fid.writelines(data)
            else:
                # 拼一行 images.txt 的内容，'\n\n'表示images.txt中一张图一般对应两行
                # 假设当前：
                # idx = 0, img_name = "IMG_0005.JPG"
                # 且：
                # images[img_name] = [
                #     "0.99", "0.01", "0.02", "0.03",
                #     "1.0", "2.0", "3.0",
                #     "1",
                #     "IMG_0005.JPG"
                # ]
                # 写出去后，文件里这一段就会变成：
                # 1 0.99 0.01 0.02 0.03 1.0 2.0 3.0 1 IMG_0005.JPG
                data = [str(1 + idx)] + [' ' + item for item in images[img_name]] + ['\n\n']
                fid.writelines(data)
    # 在相机位姿已知的前提下，用 database.db 里的特征和匹配，把 12_views/images 这批图重新三角化、BA，生成新的 sparse 重建结果
    os.system('colmap point_triangulator --database_path database.db --image_path images --input_path created  --output_path sparse  --Mapper.ba_local_max_num_iterations 40 --Mapper.ba_local_max_refinements 3 --Mapper.ba_global_max_num_iterations 100')

    if flag:
        os.system('cp sparse/points3D.bin' + f'  ../sparse/0/points3D_{n_views}views_aug.bin')
    else:
        os.system('cp sparse/points3D.bin' + f'  ../sparse/0/points3D_{n_views}views.bin')

    # 总结：created文件夹下 cameras.txt 是复制的 原sparse里面的，images.txt 由 原sparse拼接而成，points3D.bin 由 三角化直接得出

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parameters")
    parser.add_argument('--base_path', type=str, required=True, help="Path to the scene directory.")
    parser.add_argument('--views', type=int, default=3, help="The number of training views.")
    parser.add_argument('--scene', type=str, required=True, help="Scene name to process.")
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()

    pipeline(scene=args.scene, base_path=args.base_path, n_views=args.views, flag=args.augment)
