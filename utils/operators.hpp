#ifndef UTILS_OPERATORS_HPP
#define UTILS_OPERATORS_HPP

template <class T>
struct equality_operators {
    /* 
    ** The deriving class must implement the following:
    ** friend bool operator==(const T&, const T&);
    */

    friend bool operator!=(const T& lhs, const T& rhs) { return !(lhs == rhs); }
};

template <class T>
struct less_than_operators {
    /* 
    ** The deriving class must implement the following:
    ** friend bool operator<(const T&, const T&);
    */

    friend bool operator>(const T& lhs, const T& rhs)  { return rhs < lhs; }
    friend bool operator<=(const T& lhs, const T& rhs) { return !(lhs > rhs); }
    friend bool operator>=(const T& lhs, const T& rhs) { return !(lhs < rhs); }
};

template <class T>
struct relational_operators : equality_operators<T>, less_than_operators<T> { };

#endif /* UTILS_OPERATORS_HPP */