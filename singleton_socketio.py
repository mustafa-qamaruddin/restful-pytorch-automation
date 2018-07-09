from flask_socketio import SocketIO, emit
from rq.registry import StartedJobRegistry


socketio = None


def init_socket(app):
    global socketio
    socketio = SocketIO(app)
    return socketio


def push(channel, data, group):
    # @todo if connected only
    global socketio
    socketio.emit(channel,
                  data,
                  namespace=group
                  )


def background_thread(q, redis_conn):
    """Example of how to send server generated events to clients."""
    while True:
        socketio.sleep(1)

        registry = StartedJobRegistry('default', connection=redis_conn)

        # running_job_ids = registry.get_job_ids()  # Jobs which are exactly running.
        # expired_job_ids = registry.get_expired_job_ids()
        # print(registry.get_expired_job_ids())

        jobs = []
        jobs.extend(q.job_ids)
        jobs.extend(registry.get_job_ids())
        # jobs.extend(registry.get_expired_job_ids())

        for jid in jobs:
            job = q.fetch_job(jid)

            if job.meta.get('progress'):
                dict_data = {
                    "job_id": jid,
                    "job_status": job.status,
                    "job_progress": job.meta["progress"]
                }
                try:
                    push('my_response',
                         dict_data,
                         '/test')
                except:
                    pass
            elif job.status == "queued":
                dict_data = {
                    "job_id": jid,
                    "job_status": job.status
                }

                push('my_response',
                     dict_data,
                     '/test')
