from juggernaut import Juggernaut


jug = Juggernaut()


def send_notifications():
    """Notifies clients about new pastes."""
    data = {'paste_id': 10, 'reply_id': 20, 'user': 25}
    jug.publish('job-status:%d' % 200, data)


send_notifications()
