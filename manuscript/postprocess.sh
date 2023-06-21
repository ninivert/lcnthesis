#!/bin/sh
# fix broken 404 page
cd ../docs/ && sed -i 's/\/site_libs/site_libs/g' 404.html