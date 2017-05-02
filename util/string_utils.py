def xuni(s):
    if not s:
        return u''
    elif type(s) == str:
        return unicode(s, 'utf-8')
    elif type(s) == unicode:
        return s.encode('utf-8')
    else:
        return u''

def xstr(s):
    if not s:
        return ''
    elif type(s) == unicode:
        return s.encode('ascii','replace')
    else:
        return ''

