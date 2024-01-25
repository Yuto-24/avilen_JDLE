import os

from pprint import pprint
from shutil import copy
from os.path import dirname, abspath, join

dir_path = dirname(abspath("__file__"))
code_path = join(dir_path, "3_コーディング演習")
copy_path = join(dir_path, "3_コーディング演習", "提出")
compile_path = join(dir_path, "3_コーディング演習", "complete")

files = os.listdir(code_path)
files.sort()
files = list(filter(lambda x: "コーディング演習" in x and ".ipynb" in x, files))


def get_new_filepath(input_str: str):
    # copy 先 dir にあり、 Chapter[input_num] におけるファイル一覧を取得
    copy_dir_files = list(
        filter(lambda x: f"コーディング演習Chapter{input_str}" in x, os.listdir(copy_path))
    )
    copy_dir_files.sort()

    # 上記で取得したファイル数
    n = len(copy_dir_files)

    if n == 0:
        # ファイル数が 0 の場合
        file_num = 0.1
    else:
        # ファイル数が 1 以上の場合
        last_file = copy_dir_files[len(copy_dir_files) - 1]
        file_num = format(
            float(last_file.split("_")[1].split("v")[1].replace(".ipynb", "")) + 0.1,
            ".1f",
        )

    # コピー先ファイルパスを設定
    new_filename = f"コーディング演習Chapter{input_str}（植松悠杜）_v{file_num}.ipynb"
    new_filepath = join(copy_path, new_filename)
    return new_filepath


def main(input_str: str, is_complete: bool = False):
    str_num = str(input_str)

    # input_str が一桁の時、0詰め
    if len(str_num) == 1:
        str_num = "0" + str_num

    original_filepath = join(code_path, f"コーディング演習Chapter{str_num}.ipynb")

    if not is_complete:
        new_filepath = get_new_filepath(str_num)
    else:
        new_filename = f"コーディング演習Chapter{str_num}（植松悠杜）_v1.0.ipynb"
        new_filepath = join(compile_path, new_filename)

    print(f"copy from and to\n\t{original_filepath}\n\t{new_filepath}")
    copy(original_filepath, new_filepath)


# main(2, True)
# main(3)
# main("11-1")
