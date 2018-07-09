cmake_minimum_required(VERSION 2.8.12)

install(DIRECTORY ./maum/brain/mrc
    DESTINATION lib/python/maum/brain
    FILES_MATCHING PATTERN "*.py")
