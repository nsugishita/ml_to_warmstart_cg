#include <vector>
#include <iostream>

#include <gtest/gtest.h>

#include "uniquelist/unique_list.h"

TEST(TestUtilsUniqueList, TestUniqueList) {
  {
    using T = double;
    uniquelist::unique_array_list<T, uniquelist::strictly_less> list(3);

    {
      std::vector<T> a = {2.9, -1.0, 4.9};
      auto [pos, isnew] = list.push_back(a.data());
      EXPECT_EQ(pos, 0);
      EXPECT_EQ(isnew, 1);
    }

    {
      std::vector<T> a = {3.4, 1.0, 4.9};
      auto [pos, isnew] = list.push_back(a.data());
      EXPECT_EQ(pos, 1);
      EXPECT_EQ(isnew, 1);
    }

    {
      std::vector<T> a = {5.5, 5.0, 0.0};
      auto [pos, isnew] = list.push_back(a.data());
      EXPECT_EQ(pos, 2);
      EXPECT_EQ(isnew, 1);
    }

    {
      std::vector<T> a = {3.4, 1.0, 4.8999999999};
      auto [pos, isnew] = list.push_back(a.data());
      EXPECT_EQ(pos, 1);
      EXPECT_EQ(isnew, 0);
    }

    {
      std::vector<T> a = {5.5, 5.0, 0.0};
      auto it = std::begin(list);
      ++it;
      auto [pos, isnew] = list.insert(it, a.data());
      EXPECT_EQ(pos, 2);
      EXPECT_EQ(isnew, 0);
    }

    {
      std::vector<T> a = {1.5, 1.0, 0.1};
      auto it = std::begin(list);
      ++it;
      auto [pos, isnew] = list.insert(it, a.data());
      EXPECT_EQ(pos, 1);
      EXPECT_EQ(isnew, 1);
    }

    {
      std::vector<T> a = {5.5, 5.0, 0.0};
      auto result = list.isin(a.data());
      EXPECT_EQ(result, 1);
    }

    {
      std::vector<T> a = {1.5, 1.0, 0.1};
      auto result = list.isin(a.data());
      EXPECT_EQ(result, 1);
    }

    {
      std::vector<T> a = {1.5, 1.4, 4.0};
      auto result = list.isin(a.data());
      EXPECT_EQ(result, 0);
    }

    EXPECT_EQ(std::size(list), 4);

    std::vector<int> flags = {false, true};
    list.erase_nonzero(std::size(flags), flags.data());

    EXPECT_EQ(std::size(list), 3);
  }
}
