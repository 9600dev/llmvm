import requests
import urllib
import inspect
import pprint
import datetime

hn_url = "http://hn.algolia.com/api/v1/"
item_url = "{}items/".format(hn_url)
user_url = "{}users/".format(hn_url)
search_url = "{}search".format(hn_url)
date_url = "{}_by_date".format(search_url)

pp = pprint.PrettyPrinter(indent=4)

max_hits_per_page = 1000


def attr_list(obj):
    members = inspect.getmembers(obj, lambda a: not (inspect.isroutine(a)))
    return [m[0] for m in members if not (m[0].startswith("__"))]


class QueryFailed(BaseException):
    """ Raised when the server request fails. """


class Hit(object):
    """basic hit object returned by search"""

    def __init__(self, **fields):
        for key in fields.keys():
            setattr(self, key, fields[key])

    def __repr__(self):
        return pp.pformat(self.json())

    def json(self):
        return {attr: self.__dict__[attr] for attr in attr_list(self)}

    @classmethod
    def get_type_cls_from_fields(cls, fields):
        type_map = {
            "story": Story,
            "poll": Poll,
            "pollopt": PollOption,
            "comment": Comment,
            "user": User,
        }
        if "_tags" in fields:
            type_key = list(type_map.keys() & fields["_tags"])[0]
            return type_map[type_key]
        if "type" in fields:
            return type_map[fields["type"]]
        if "username" in fields:
            return User
        return cls

    @classmethod
    def make(cls, fields):
        type_cls = cls.get_type_cls_from_fields(fields)
        return type_cls(**fields)

    # 'public' instance methods - use these in your code

    def get_full(self):
        """get by id gives more information than a search list item for some types"""
        if hasattr(self, "objectID"):
            return SearchHN().item(self.objectID).get()

    def get_parent_object(self):
        if hasattr(self, "parent_id"):
            if self.parent_id:
                return SearchHN().item(self.parent_id).get()


class Story(Hit):
    def get_author(self):
        return SearchHN().user(self.author).get()

    def get_story_comments(self):
        return SearchHN().story(self.objectID).comments().max_hits_per_page().get()


class Poll(Story):
    def get_poll_options(self):
        # TODO this doesn't work because pollopts don't have a parent ref
        # return SearchHN().story(self.objectID).poll_options().get()
        raise NotImplementedError


class PollOption(Hit):
    def get_parent_poll(self):
        return SearchHN().item(self.parent_id).get()
        pass


class Comment(Hit):
    def get_parent_story(self):
        return SearchHN().item(self.story_id).get()


class User(Hit):
    def get_full(self):
        if hasattr(self, "username"):
            return SearchHN().user(self.username).get()

    def get_user_comments(self):
        return SearchHN().author(self.username).comments().get()

    def get_user_stories(self):
        return SearchHN().author(self.username).stories().get()


class SearchHN(object):
    """creates and executes the query"""

    def __init__(self):
        self.param_obj = {}
        self.base_url = search_url
        self.single_item = False
        self.full_url = None

    def __repr__(self):
        return "SearchHN object:\n{}".format(pp.pformat(self._json()))

    def _json(self):
        return {attr: self.__dict__[attr] for attr in attr_list(self)}

    def _add_numeric_filter(self, filter):
        if "numericFilters" not in self.param_obj:
            self.param_obj["numericFilters"] = []
        self.param_obj["numericFilters"].append(filter)
        return self

    def _created_at_i(self, symbol, ts):
        if type(ts) == datetime.datetime:
            ts = ts.timestamp()
        created = "created_at_i{}{}".format(symbol, ts)
        return self._add_numeric_filter(created)

    def _add_tag(self, tag):
        if "tags" not in self.param_obj:
            self.param_obj["tags"] = []
        self.param_obj["tags"].append(tag)
        return self

    def _single(self):
        """ Get single item/user by id/username, not searching. """
        self.single_item = True
        return self

    def _add_request_fields(self, json):
        """ Add request's response fields after executing a search. """
        for key in [k for k in json.keys() if k != "hits"]:
            setattr(self, key, json[key])

    def _get_field_str(self, field):
        """ Because requests.get() does this wrong. """
        result = ""
        if field in self.param_obj:
            result = "&{}=".format(field) + ",".join(self.param_obj[field])
            self.param_obj.pop(field, None)
        return result

    def _get_full_url(self):
        tags_str = self._get_field_str("tags")
        numeric_str = self._get_field_str("numericFilters")
        param_str_minus_tags = urllib.parse.urlencode(self.param_obj)
        return "{}?{}{}{}".format(
            self.base_url, param_str_minus_tags, numeric_str, tags_str
        )

    def _request(self):
        full_url = self._get_full_url()
        setattr(self, "full_url", full_url)
        return requests.get(full_url)

    # "public" methods - use these
    # design choice: should methods that don't take an arg besides self
    # have @property? Currently do not, for the sake of consistency

    def search(self, query_str):
        self.param_obj["query"] = query_str
        return self

    def min_points(self, points):
        return self._add_numeric_filter("points>={}".format(points))

    def min_comments(self, comments):
        return self._add_numeric_filter("num_comments>={}".format(comments))

    def latest(self):
        self.base_url = date_url
        return self

    def created_after(self, timestamp):
        return self._created_at_i(">", timestamp)

    def created_before(self, timestamp):
        return self._created_at_i("<", timestamp)

    def created_between(self, ts1, ts2):
        self.created_after(ts1)
        return self.created_before(ts2)

    def stories(self):
        return self._add_tag("story")

    def comments(self):
        return self._add_tag("comment")

    def polls(self):
        return self._add_tag("poll")

    def poll_options(self):
        return self._add_tag("pollopt")

    def author(self, author):
        return self._add_tag("author_{}".format(author))

    def whoishiring_threads(self):
        return self.author("whoishiring").stories().search("hiring")

    def whowantstobehired_threads(self):
        return self.author("whoishiring").stories().search("hired")

    def story(self, story_id):
        return self._add_tag("story_{}".format(story_id))

    def item(self, object_id):
        self.base_url = item_url + str(object_id)
        return self._single()

    def user(self, username):
        self.base_url = user_url + username
        return self._single()

    def hits_per_page(self, num_hits):
        self.param_obj["hitsPerPage"] = num_hits
        return self

    def max_hits_per_page(self):
        return self.hits_per_page(max_hits_per_page)

    def page(self, page_num):
        self.param_obj["page"] = page_num
        return self

    def reset(self):
        self = self.__init__()
        return self

    def get(self, reset=True):
        """
        `reset` as kwarg because a user may want to use same query but
        increment page count.
        """
        resp = self._request()
        if not resp.ok:
            raise QueryFailed
        if self.single_item:
            result = Hit.make(resp.json())
        else:
            result = [Hit.make(hit) for hit in resp.json()["hits"]]
            self._add_request_fields(resp.json())
        if reset:
            self.reset()
        return result

    def get_first(self):
        return self.hits_per_page(1).get()[0]

    # non composable - quick and easy results

    def get_item(self, item):
        return self.item(item).get()

    def get_user(self, username):
        return self.user(username).get()

    def get_latest_stories(self):
        return self.latest().stories().get()

    def search_stories(self, query):
        return self.search(query).get()

    def get_latest_comments(self):
        return self.latest().comments().get()

    def search_comments(self, query):
        return self.search(query).comments().get()

    def get_latest_whoishiring_thread(self):
        return self.whoishiring_threads().latest().get_first()

    def get_latest_whowantstobehired_thread(self):
        return self.whowantstobehired_threads().latest().get_first()


if __name__ == "__main__":
    # TODO make cli
    hn = SearchHN()
    print(hn.latest().stories().get_first())
