"""
Author: maggie
Date:   2021-06-15
Place:  Xidian University
@copyright
"""

import click

class CommaSeparatedList(click.ParamType):
    name = 'list'
    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')