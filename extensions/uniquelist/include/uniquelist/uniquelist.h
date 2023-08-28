/**
 * @file
 * @author Nagisa Sugishita <s1576972@ed.ac.uk>
 * @version 1.0
 *
 * List to keep unique elements.
 *
 * An extension of std::list which only takes unique elements.
 */

#ifndef UNIQUELIST_UNIQUELIST_H
#define UNIQUELIST_UNIQUELIST_H

#include <list>
#include <map>
#include <memory>  // std::shared_ptr
#include <utility> // std::pair

namespace uniquelist {

/**
 * @brief Linked list which only keeps unique elements
 *
 * This is a linked list which only contains unique elements.
 * Elements in this list shall not be modified after they
 * are added until they are removed from the list.
 *
 * Internally, this uses map to sort elements so that
 * membership of an element can be tested quickly.
 * The way to sort elements can be specified by Compare.
 */
template <typename T, typename Compare = std::less<T>> struct uniquelist {

private:
  struct map_item_type;

  /**
   * @brief Type of items added in the underlying list
   *
   * Instances of this struct are added to the underlying list.
   * Each instance has an iterator to an item in the map
   * and that item in the map has an iterator pointint back
   * to the original element in the list.
   */
  struct list_item_type {
    using link_type = typename std::map<T, map_item_type, Compare>::iterator;

    link_type link;
  };

  /**
   * @brief Type of items added in the underlying map
   *
   * This type is used as a value_type for the underlying map.
   * Each instance has an iterator to an item in the list
   * and that item in the list has an iterator pointint back
   * to the original element (and its key) in the map.
   */
  struct map_item_type {
    using link_type = typename std::list<list_item_type>::iterator;

    link_type link;
  };

  /**
   * @brief Type of the underlying list
   */
  using list_type = std::list<list_item_type>;

  /**
   * @brief Type of the underlying map
   */
  using map_type = std::map<T, map_item_type, Compare>;

  /**
   * @brief Iterator to iterate elements in the order they are added
   *
   * This is a bidirectional iterator to iterate over
   * elements in the order they are added.
   *
   * The template parameter S must be an iterator type
   * of list_type.  This keeps an iterator of the list_type
   * internally * and surrogate all operations.  When an
   * iterator is dereferenced, the key of the corresponding
   * map elements are returned.
   */
  template <typename S> struct iterator_template {
    using iterator_category = typename S::iterator_category;
    using value_type = S;
    using difference_type = typename S::difference_type;
    using pointer = value_type *;
    using reference = value_type &;

    explicit iterator_template(const S &it) noexcept : it{it} {}

    auto &operator++() noexcept {
      ++it;
      return *this;
    }

    auto &operator--() noexcept {
      --it;
      return *this;
    }

    auto operator++(int) noexcept {
      auto buf = *this;
      ++*this;
      return buf;
    }

    auto operator--(int) noexcept {
      auto buf = *this;
      --*this;
      return buf;
    }

    template <typename U> auto operator==(const U &other) const noexcept {
      return it == other.it;
    }

    template <typename U> auto operator!=(const U &other) const noexcept {
      return !(*this == other);
    }

    const auto &operator*() { return it->link->first; }

    const auto &operator*() const { return it->link->first; }

    const auto *operator->() { return &it->link->first; }

    const auto *operator->() const { return &it->link->first; }

    auto unwrap() noexcept { return it; }

  private:
    S it;
  };

public:
  using value_type = T;
  using reference = value_type &;
  using const_reference = const value_type &;
  using pointer = value_type *;
  using const_pointer = const value_type *;
  using iterator = iterator_template<typename list_type::iterator>;
  using const_iterator = iterator_template<typename list_type::const_iterator>;

  /* Member functions */

  /* Iterators */

  /**
   * @brief Returns an iterator pointing to the first element
   *
   * @return An iterator to the beginning of the sequence container.
   */
  auto begin() noexcept { return iterator(std::begin(list)); }

  /**
   * @brief Returns an iterator pointing to the first element
   *
   * @return An iterator to the beginning of the sequence container.
   */
  auto begin() const noexcept { return const_iterator(std::begin(list)); }

  /**
   * @brief Returns an iterator referring to the past-the-end element
   *
   * @return An iterator to the element past the end of the sequence.
   */
  auto end() noexcept { return iterator(std::end(list)); }

  /**
   * @brief Returns an iterator referring to the past-the-end element
   *
   * @return An iterator to the element past the end of the sequence.
   */
  auto end() const noexcept { return const_iterator(std::end(list)); }

  /* Capacity */

  /**
   * @brief Test whether container is empty
   *
   * @return true if the container size is 0, false otherwise.
   */
  auto empty() const noexcept {
    return list.empty();
  }

  /**
   * @brief Return size
   *
   * @return The number of elements in the container.
   */
  auto size() const noexcept { return list.size(); }

  /**
   * @brief Return maximum size
   *
   * @return The maximum number of elements the object can hold as content.
   */
  auto max_size() const noexcept {
    return std::min(list.max_size(), map.max_size());
  }

  /* Modifiers */

  /**
   * @brief Add a new item to the end if it is not in the list
   *
   * @param [in] Value to be copied (or moved) to the new element.
   *
   * @return Pair of the position of the given item in the list
   *     and status.  status = true indicates that the item is added
   *     as a new one and false indicates that the item is already
   *     in the list.
   */
  auto push_back(const T &key) { return insert(std::end(*this), key); }

  /**
   * @brief Add a new item to the end if it is not in the list
   *
   * This tries to push_back a new element to the list.  If it
   * is already included, this does not have any effect
   * (and the argument remains valid).
   *
   * @param [in] Value to be copied (or moved) to the new element.
   *
   * @return Pair of the position of the given item in the list
   *     and status.  status = true indicates that the item is added
   *     as a new one and false indicates that the item is already
   *     in the list.
   */
  auto push_back(T &&key) { return insert(std::end(*this), std::move(key)); }

  /**
   * @brief Add a new item to the end if it is not in the list
   *
   * This tries to push_back a new element to the list.
   * If the given element is not in the list, the data is
   * copied and added to the list.
   * If it is already included, this does not have any effec
   * (data is not copied).
   *
   * @param [in] Value to be copied to the new element.
   *
   * @return Pair of the position of the given item in the list
   *     and status.  status = true indicates that the item is added
   *     as a new one and false indicates that the item is already
   *     in the list.
   */
  template <typename F> auto push_back_with_hook(const T &key, const F &f) {
    return insert_with_hook(std::end(*this), key, f);
  }

  /**
   * @brief Insert a new element before the the specified position
   *
   * This tries to insert a new element to the list.
   * If the given item is already in the list, this does not have
   * any effects.
   *
   * @param [in] Position in the container where
   *     the new elements are inserted.
   * @param [in] Value to be copied to the new element.
   *
   * @return Pair of the position of the given item in the list
   *     and status.  status = true indicates that the item is added
   *     as a new one and false indicates that the item is already
   *     in the list.
   */
  template <typename S>
  auto insert(iterator_template<S> position, const value_type &val) {
    auto [it, status] = map.insert(std::pair(val, map_item_type{}));
    if (status) {
      it->second.link = list.insert(position.unwrap(), list_item_type{it});
    }
    return std::pair<size_t, bool>(
        std::distance(std::begin(list), it->second.link), status);
  }

  /**
   * @brief Insert a new element before the the specified position
   *
   * This tries to insert a new element to the list.
   * If the given item is already in the list, this does not have
   * any effects (and the argument remains valid).
   *
   * @param [in] Position in the container where
   *     the new elements are inserted.
   * @param [in] Value to be copied to the new element.
   *
   * @return Pair of the position of the given item in the list
   *     and status.  status = true indicates that the item is added
   *     as a new one and false indicates that the item is already
   *     in the list.
   */
  template <typename S>
  auto insert(iterator_template<S> position, value_type &&val) {
    auto [it, status] = map.try_emplace(std::move(val), map_item_type{});
    if (status) {
      it->second.link = list.insert(position.unwrap(), list_item_type{it});
    }
    return std::pair<size_t, bool>(
        std::distance(std::begin(list), it->second.link), status);
  }

  /**
   * @brief Insert a new item to the end if it is not in the list
   *
   * This tries to insert a new element to the list.
   * If the given element is not in the list, the data is
   * copied and added to the list.
   * If it is already included, this does not have any effec
   * (data is not copied).
   *
   * @param [in] Position in the container where
   *     the new elements are inserted.
   * @param [in] Value to be copied to the new element.
   *
   * @return Pair of the position of the given item in the list
   *     and status.  status = true indicates that the item is added
   *     as a new one and false indicates that the item is already
   *     in the list.
   */
  template <typename S, typename F>
  auto insert_with_hook(iterator_template<S> position, const value_type &val,
                        const F &f) {
    auto [try_it, try_status] = map.insert(std::pair(val, map_item_type{}));
    if (try_status) { // If the given item is new.
      // The shallowcopy is inserted as a new item.
      // Remember the place, remove the copy and
      // re-insert a deepcopy.
      auto hint = try_it; // Pos just before the newly added element.
      ++hint;
      map.erase(try_it); // Remove the shallow copy.
      // Creat a deepcopy and re-insert to the map.
      auto it = map.emplace_hint(hint, f(val), map_item_type{});
      it->second.link = list.insert(position.unwrap(), list_item_type{it});
      return std::pair<size_t, bool>(
          std::distance(std::begin(list), it->second.link), try_status);
    } else { // If the given item is not new.
      return std::pair<size_t, bool>(
          std::distance(std::begin(list), try_it->second.link), try_status);
    }
  }

  /**
   * @brief Remove an element
   *
   * This removes an element pointed by a given iterator.
   *
   * @param [in] it Iterator pointint to an element to be removed
   *
   * @return An iterator pointint to the element that followed
   *     the element erased by the function call.
   */
  auto erase(iterator it) { return iterator(erase_by_list_it(it.unwrap())); }

  /**
   * @brief Remove an element
   *
   * This removes an element pointed by a given iterator.
   *
   * @param [in] it Iterator pointint to an element to be removed
   *
   * @return An iterator pointint to the element that followed
   *     the element erased by the function call.
   */
  auto erase(const_iterator it) {
    return const_iterator(erase_by_list_it(it.unwrap()));
  }

  /**
   * @brief Erase an element at a given position
   *
   * This removes an element at a given position.
   *
   * @param [in] index Index of element to be removed.
   *
   * @return An iterator pointint to the element that followed
   *     the element erased by the function call.
   */
  auto erase(size_t index) {
    auto it = std::begin(*this);
    std::advance(it, index);
    return erase(it);
  }

  /**
   * @brief Erase elements at given positions
   *
   * Remove elements at given positions.  `indexes` must be sorted
   * in the increasing order.
   *
   * @param [in] n Number of elements to be removed
   * @param [in] indexes Array of indexes of elements to be removed.
   *     The indexes must be sorted in the increasing order.  size: n
   */
  template <typename U> auto erase(size_t n, const U *indexes) {
    auto prev_pos = 0;
    auto cursor = ++std::begin(*this);
    for (size_t i = 0; i < n; ++i) {
      std::advance(cursor, indexes[i] - prev_pos - 1);
      cursor = erase(cursor);
      prev_pos = indexes[i];
    }
  }

  /**
   * @brief Erase elements at the positions of nonzero elements
   *
   * Remove elements at the positions of nonzero elements.
   * Elements which are evaluated as true is considered
   * to be nonzero.
   *
   * @param [in] n Size of `flags`
   * @param [in] flag Array of flags whose nonzero elements
   *     indicate the removal of the corresponding elements.  size: n
   */
  template <typename U> auto erase_nonzero(size_t n, const U *flag) {
    auto cursor = std::begin(*this);
    for (size_t i = 0; i < n; ++i) {
      if (flag[i]) {
        cursor = erase(cursor);
      } else {
        ++cursor;
      }
    }
  }

  /**
   * @brief Remove all elements
   */
  auto clear() noexcept {
    list.clear();
    map.clear();
  }

  /**
   * @brief Test if the given item is in the list or not
   *
   * @param [in] val Value to search for.
   *
   * @return true if contained and 0 otherwise.
   */
  auto isin(const T &val) const noexcept { return map.count(val) > 0; }

private:
  /**
   * @brief Actual list to maintain elements.
   *
   * This keeps all elements in this uniquelist.
   * More precisely, all elements are kept as keys
   * in the map and this list contains iterators
   * to the key, value pairs of map.  The order
   * of elements in this list is the same as
   * the order of the elements added, and traversing
   * this list allows us to travese elements in
   * the insertion order.
   */
  list_type list{};

  /**
   * @brief Map to maintain given elements in the sorted order
   *
   * This keeps all given elements as keys.  The corresponding
   * values are iterators to the elements in list.  From
   * a key, value pair of this map, one can use the iterator
   * to get the element in the list, and compute the index
   * based on their insertion, say.
   */
  map_type map{};

  /**
   * @brief Remove an element
   *
   * This removes an element pointed by a given iterator.
   *
   * @param [in] it Iterator pointint to an element to be removed
   */
  auto erase_by_list_it(typename list_type::iterator it) {
    map.erase(it->link);
    return list.erase(it);
  }

  /**
   * @brief Remove an element
   *
   * This removes an element pointed by a given iterator.
   *
   * @param [in] it Iterator pointint to an element to be removed
   */
  auto erase_by_list_it(typename list_type::const_iterator it) {
    map.erase(it->link);
    return list.erase(it);
  }

}; // struct uniquelist

/**
 * @brief Deleter for a view
 *
 * This is a deleter which does not release any data.
 * This may be used for a shared_ptr which is a view of a pointer.
 */
template <typename T> struct no_delete {
  void operator()(T *p) const { (void)p; }
};

template <typename T> struct no_delete<T[]> {
  void operator()(T *p) const { (void)p; }
};

template <typename I, typename T>
auto deepcopy(const std::pair<I, std::shared_ptr<T[]>> &a) {
  using T_ = std::remove_const_t<T>;
  std::shared_ptr<T_[]> p(new T_[a.first]);
  std::copy(a.second.get(), a.second.get() + a.first, p.get());
  return std::pair<I, std::shared_ptr<T[]>>{a.first, p};
}

template <typename T> std::shared_ptr<T[]> shared_ptr_without_ownership(T *p) {
  return std::shared_ptr<T[]>(p, no_delete<T[]>());
}

template <typename Compare> struct array_less {

  template <typename I, typename T>
  auto operator()(const std::pair<I, std::shared_ptr<T[]>> &a,
                  const std::pair<I, std::shared_ptr<T[]>> &b) const {
    if (a.first != b.first) {
      return a.first < b.first;
    }
    Compare compare{};
    auto ap = a.second.get();
    auto bp = b.second.get();
    for (I i = 0; i < a.first; ++i) {
      if (compare(ap[i], bp[i])) {
        return true;
      } else if (compare(bp[i], ap[i])) {
        return false;
      }
    }
    return false;
  }
};

template <typename T, typename Compare>
using unique_array_list_super_class =
    uniquelist<std::pair<size_t, std::shared_ptr<const T[]>>,
                array_less<Compare>>;

template <typename T, typename Compare = std::less<T>>
struct unique_array_list
    : unique_array_list_super_class<T, Compare> {
  using uniquelist_ =
      uniquelist<std::pair<size_t, std::shared_ptr<const T[]>>,
                  array_less<Compare>>;

  size_t array_size;

  unique_array_list(size_t array_size) : unique_array_list_super_class<T, Compare>{}, array_size{array_size} {}

  auto push_back(const T *key) {
    std::shared_ptr<const T[]> key_view = shared_ptr_without_ownership(key);
    auto copy_ = [] (const std::pair<size_t, std::shared_ptr<const T[]>> &a) { return deepcopy<size_t, const T>(a);};
    return unique_array_list_super_class<T, Compare>::push_back_with_hook({this->array_size, key_view},
                                             copy_);
  }

  auto insert(typename unique_array_list_super_class<T, Compare>::iterator position, const T *key) {
    std::shared_ptr<const T[]> key_view = shared_ptr_without_ownership(key);
    return unique_array_list_super_class<T, Compare>::insert_with_hook(position, {this->array_size, key_view},
                                          deepcopy<size_t, const T>);
  }

  auto isin(const T *key) const noexcept {
    std::shared_ptr<const T[]> key_view = shared_ptr_without_ownership(key);
    return unique_array_list_super_class<T, Compare>::isin({this->array_size, key_view});
  }
};

/**
 * @brief Compare two numbers with a tolerance
 */
struct strictly_less {
  double rtol;
  double atol;

  strictly_less(double rtol = 1e-6, double atol = 1e-6)
      : rtol{rtol}, atol{atol} {}

  template <typename T> bool operator()(T a, T b) const {
    return a < b - ((b > 0) ? b : -b) * this->rtol - this->atol;
  }
};

} // namespace uniquelist

#endif // UNIQUELIST_UNIQUELIST_H
