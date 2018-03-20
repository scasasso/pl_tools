# -*- coding: utf-8 -*-
"""
################################################################################
#
# File:     api_handler.py
#
# Product:  Predictive Layer Core 2015
# Author:   Momo
# Date:     28 May 2015
#
# Scope:    The file contains the ApiHandler class which provide functions
#           to interoperate with PredictiveLayer APIs
#
# Copyright (c) 2015, Predictive Layer Limited.  All Rights Reserved.
#
# The contents of this software are proprietary and confidential to the author.
# No part of this program may be photocopied,  reproduced, or translated into
# another programming language without prior written consent of the author.
#
#
#
################################################################################
"""

import json
import requests
import logging

# Convention: Always use logger with file name as key
#
logger = logging.getLogger(__file__)


class ApiHandler:

    def __init__(self):
        self.s = requests.Session()
        self.host = None
        self.url = None
        return

    def set_host(self, host):
        self.host = host

    def set_url(self, url):
        self.url = url

    def push_update(self, push_dict):
        payload = {'data': push_dict}
        request = requests.Request("POST", self.url % self.host, data=json.dumps(payload))
        request = request.prepare()
        request.headers['Content-Type'] = 'application/json'
        r = self.s.send(request, verify=False)

        if r.status_code != 200:
            logger.warning('Fails to push data: %s Error code: %s' % (json.dumps(payload), r.status_code))
            logger.warning(r.text)
        else:
            logger.info('Loading URL with data OK.')
        return

    def push_delete(self, datetime):
        link = self.url + '&date=' + datetime
        r = self.s.delete(link)
        if r.status_code != 200:
            logger.warning('Fails to delete the entry for date: %s. Error code: %s' % (datetime, r.status_code))
            logger.warning(r.text)
        else:
            logger.info('Deleting the entry is ok. Thanks')
        return
