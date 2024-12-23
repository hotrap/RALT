namespace ralt {
// template <typename Key, void (*Update)(Key&, const Key&, const Key&), void (*SoloUpdate)(Key&, const Key&), void (*EmptyUpdate)(Key&),
//           void (*Union)(Key&, const Key&), int (*Compare)(const Key&, const Key&)>
template <typename Key, typename UpdateT, typename SoloUpdateT, typename EmptyUpdateT, typename DupUpdateT, typename CompareT>
class Splay {
  UpdateT Update;
  SoloUpdateT SoloUpdate;
  EmptyUpdateT EmptyUpdate;
  DupUpdateT DupUpdate;
  CompareT Compare;

 public:
  struct NodeData {
    Key key;
    NodeData *fa, *lc, *rc;
    NodeData *prv, *nxt;
  };
  Splay(const UpdateT& _Update, const SoloUpdateT& _SoloUpdate, const EmptyUpdateT& _EmptyUpdate, const DupUpdateT& _DupUpdate, const CompareT& _Compare)
      : Update(_Update), SoloUpdate(_SoloUpdate), EmptyUpdate(_EmptyUpdate), DupUpdate(_DupUpdate), Compare(_Compare), rt_(nullptr) {}
  ~Splay() {
    _dfs_delete(rt_);
  }
  
  void insert(const Key& key) {
    auto prv = _prev(key);
    if (prv != nullptr) _splay(prv);
    if (prv != nullptr && Compare(prv->key, key) == 0) {
      DupUpdate(prv->key, key);
      return;
    }
    auto nxt = _next(key);
    auto node = new NodeData({key, nullptr, nullptr, nullptr, prv, nxt});
    _update(node);
    assert(!prv || prv->nxt == nxt);
    assert(!nxt || nxt->prv == prv);
    assert(!prv || Compare(prv->key, key) < 0);
    assert(!nxt || Compare(nxt->key, key) > 0);
    if (prv) prv->nxt = node;
    if (nxt) nxt->prv = node;
    if (prv && !prv->rc) {
      node->fa = prv;
      prv->rc = node;
      _update(prv);
    } else if (nxt && !nxt->lc) {
      node->fa = nxt;
      nxt->lc = node;
      _update(nxt);
    } else {
      rt_ = node;
    }
    _splay(node);
  }
  void erase(const Key& key) {
    auto x = _prev(key);
    if (x != nullptr) _splay(x);
    if (x == nullptr || Compare(x->key, key) != 0) return;
    if (x->prv) x->prv->nxt = x->nxt;
    if (x->nxt) x->nxt->prv = x->prv;
    if (x->lc == nullptr && x->rc == nullptr) {
      rt_ = nullptr;
    } else if (x->lc == nullptr) {
      rt_ = x->rc;
      rt_->fa = nullptr;
    } else if (x->rc == nullptr) {
      rt_ = x->lc;
      rt_->fa = nullptr;
    } else {
      rt_ = x->lc;
      rt_->fa = nullptr;
      auto rt_r = rt_;
      while (rt_r->rc) rt_r = rt_r->rc;
      _splay(rt_r);
      rt_r->rc = x->rc;
    }
    delete x;
  }

  NodeData* upper(const Key& key) {
    auto x = _next_eq(key);
    if (x != nullptr) _splay(x);
    return x;
  }

  NodeData* begin() {
    auto x = rt_;
    if (!x) return nullptr;
    while (x->lc) x = x->lc;
    _splay(x);
    return x;
  }

  Key* presum(const Key& key) {
    if (rt_ == nullptr) return nullptr;
    auto x = _next_eq(key);
    if (x == nullptr) {
      return &rt_->key;
    } else {
      _splay(x);
      if (x->lc == nullptr) return nullptr;
      return &x->lc->key;
    }
  }

  CompareT comp() { return Compare; }

  // auto& print(NodeData* x, std::string s = "") {
  //   if (!x) return std::cout;
  //   print(x->lc, s + "->lc") << "[" << x->key.key << ", " << s << "]";
  //   return print(x->rc, s + "->rc");
  // }

 private:
  NodeData* rt_;
  void _splay(NodeData* x) {
    while (x->fa) _rotate(x);
    _update(x);
    rt_ = x;
  }
  void _rotate(NodeData* x) {
    auto fa = x->fa;
    if (x == fa->lc) {
      fa->lc = x->rc;
      if (x->rc) x->rc->fa = fa;
      x->rc = fa;
    } else {
      fa->rc = x->lc;
      if (x->lc) x->lc->fa = fa;
      x->lc = fa;
    }
    if (fa->fa) {
      auto ffa = fa->fa;
      if (ffa->lc == fa)
        ffa->lc = x;
      else
        ffa->rc = x;
    }
    x->fa = fa->fa;
    fa->fa = x;
    _update(fa);
  }
  void _update(NodeData* x) {
    if (!x->lc && !x->rc)
      EmptyUpdate(x->key);
    else if (!x->lc)
      SoloUpdate(x->key, x->rc->key);
    else if (!x->rc)
      SoloUpdate(x->key, x->lc->key);
    else
      Update(x->key, x->lc->key, x->rc->key);
  }
  NodeData* _find(const Key& k) {
    auto ret = rt_;
    while (ret) {
      auto result = Compare(k, ret->key);
      if (result == 0)
        break;
      else if (result < 0)
        ret = ret->lc;
      else
        ret = ret->rc;
    }
    return ret;
  }
  NodeData* _next(const Key& k) {
    auto x = rt_;
    NodeData* ret = nullptr;
    while (x) {
      auto result = Compare(k, x->key);
      if (result < 0)
        ret = x, x = x->lc;
      else
        x = x->rc;
    }
    return ret;
  }
  NodeData* _next_eq(const Key& k) {
    auto x = rt_;
    NodeData* ret = nullptr;
    while (x) {
      auto result = Compare(k, x->key);
      if (result <= 0)
        ret = x, x = x->lc;
      else
        x = x->rc;
    }
    return ret;
  }
  NodeData* _prev(const Key& k) {
    auto x = rt_;
    NodeData* ret = nullptr;
    while (x) {
      auto result = Compare(k, x->key);
      if (result >= 0)
        ret = x, x = x->rc;
      else
        x = x->lc;
    }
    return ret;
  }
  void _dfs_delete(NodeData* rt) {
    if(!rt) return;
    _dfs_delete(rt->lc);
    _dfs_delete(rt->rc);
    delete rt;
  }
};
}  // namespace ralt