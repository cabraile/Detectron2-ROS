cmake_minimum_required(VERSION 3.1)
project(detectron2_ros)

# Dependencies
# ------------------

find_package(catkin REQUIRED COMPONENTS rospy ros_numpy)

# Include
# -----------------

catkin_package( INCLUDE_DIRS )

include_directories( ${catkin_INCLUDE_DIRS} )

link_directories()