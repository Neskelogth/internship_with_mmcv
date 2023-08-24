from setuptools import find_packages, setup


def requirements(file):
    lines = list()
    for line in open(file, 'r'):
        lines.append(line.strip())
    return lines


if __name__ == '__main__':
    setup(
        name='poseConv',
        version='0.1.0',
        author='Samuel Kostadinov',
        author_email='samuel.kostadinov@ustudenti.unitn.it',
        keywords='Skeleton-based action recognition, computer vision',
        packages=find_packages(),
        install_requirements=requirements('requirements.txt')
    )
