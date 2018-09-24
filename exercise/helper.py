

class TestCallback:

    def __init__(self) -> None:
        self.fail_count = 0

    def start(self, name_):
        print('\n------- {} Exercise ----------\n'.format(name_))

    def start_test(self, test_name):
        print('{}:'.format(test_name), end='\t', flush=True)

    def end_test(self, passed):
        if passed:
            print('passed')
        else:
            self.fail_count += 1
            print('failed')

    def end(self):
        if self.fail_count == 0:
            print('\n------ All tests passed -------')
        else:
            print('\n------ {} tests failed. Keep on trying! --------'.format(self.fail_count))