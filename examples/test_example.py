from frozen_ground_fem.example_class import (
    TestClass,
)


def main():
    a = TestClass(
        8,
        "Anna",
        x=5.0,
        y=-3.0,
        groceries=["cranberries", "seasalt", "apples",],
        sort_groceries=True,
    )
    print(f"a.__dict__: {a.__dict__}")
    print(f"a.__doc__: {a.__doc__}")
    print(f"a.__class__: {a.__class__}")
    print(f"a.__module__: {a.__module__}")
    print(f"a.__annotations__: {a.__annotations__}")
    print(f"a.groceries: {a.groceries}")
    print(f"a.x: {a.x}")
    print(f"a.y: {a.y}")
    a.print_grocery_list()
    a.print_grocery_item(1)
    a.add_grocery_items(["bananas", "omegas"], True)
    a.print_grocery_list()


if __name__ == "__main__":
    main()
