---
layout: post
title:  "null-safe Java"
comments: true
tags:
  - Java
  - software-engineering
---

For those who have used Java, I'm sure that you have encountered a **NullPointerException (NPE)** atleast once. In this short blog post, I'm going to discuss how to be **null-safe** whenever you are dealing with these **nullable** expressions in Java.

## Java 8's Optional
Just recently, I've found out about Java 8's **Optional** wrapper. It is located in the **java.util** package.

The idea is simple: wrap **Optional** in any data type that you think might have a **null** value.

For example, you have a hash map with a key of **String** and a value of **Integer**,

```java
import java.util.*;

Map<String, Integer> map = new HashMap<>();
map.put("a", 1);
map.put("b", 2);
map.put("c", 3);
```

And, unintentionally, you try to use the **get** method of a map on a **non-existing** key,

```java
Integer dValue = map.get("d");
```

This might cause unexpected NPEs in your code. And obviously, the solution is to wrap these **nullable** values in an **Optional** wrapper.


```java
Optional<Integer> dValueOpt = Optional.ofNullable(map.get("d"));

if (dValueOpt.isPresent()) {
  Integer dValue = dValueOpt.get();
  // do something if it has a value
} else {
  // do something if it doesn't have a value
}
```

You might have to use **Optional.ofNullable** most of the time to indicate that the return value of the expression can have **null** values.

I'm sure you're wondering:

*"Why don't you just check for null values manually? Wrapping data in Optional will just yield me more keystrokes."*

```java
Integer dValue = map.get("d");

if (dvalue == null) {
  // do something if null
} else {
  // do something if NOT null
}
```

Of course, **Optional** has other methods too aside from **isPresent()** and one of the most important method is **flatMap(functionMapper)** which chains operations on an **Optional** value. Personally, I also think it's better to not use **null** in your code as much as possible. I don't know if it is just me but whenever I see **null** in my Java code, it makes me want to gouge my eye out.

And for those functional programmers out there, I think you've noticed that Java is slowly adapting functional programming concepts. I want to discuss more of these concepts but I don't have much time so that would be for another blog post and it's going to be a bit *FUN*ctional. ;)

## Conclusion

Use **Optional** if you think your data might be have **null** values.

For more details on the **Optional** wrapper, you can check the [Java 8 API documentation](https://docs.oracle.com/javase/8/docs/api/java/util/Optional.html).

Questions? Comments? Or maybe you've seen some typo? Put it in the comment section below.

