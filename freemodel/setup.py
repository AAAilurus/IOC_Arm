from setuptools import find_packages, setup

package_name = 'freemodel'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'fm_follower = freemodel.fm_follower:main',
            'fm_leader = freemodel.fm_leader:main',
            'fm_pipeline = freemodel.fm_pipeline:main',
        ],
    },
)
