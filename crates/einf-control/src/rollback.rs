/// Transaction helper: execute with exclusive `&mut Ctx`, then either
/// discard registered undos or run them LIFO with exclusive `&mut Ctx`.
pub struct RollbackScope<Ctx> {
    actions: Vec<Box<dyn FnOnce(&mut Ctx) + 'static>>,
}

impl<Ctx> RollbackScope<Ctx> {
    pub fn run<R, E>(
        ctx: &mut Ctx,
        f: impl FnOnce(&mut Self, &mut Ctx) -> Result<R, E>,
    ) -> Result<R, E> {
        let mut scope = Self {
            actions: Vec::new(),
        };
        match f(&mut scope, ctx) {
            Ok(value) => Ok(value),
            Err(error) => {
                for action in scope.actions.drain(..).rev() {
                    action(ctx);
                }
                Err(error)
            }
        }
    }

    pub fn defer(&mut self, action: impl FnOnce(&mut Ctx) + 'static) {
        self.actions.push(Box::new(action));
    }
}

#[cfg(test)]
mod tests {
    use super::RollbackScope;

    #[test]
    fn commits_on_ok() {
        let mut value = 1;
        RollbackScope::run(&mut value, |scope, ctx| {
            let previous = *ctx;
            scope.defer(move |ctx| *ctx = previous);
            *ctx = 2;
            Ok::<_, ()>(())
        })
        .unwrap();
        assert_eq!(value, 2);
    }

    #[test]
    fn rolls_back_on_err() {
        let mut value = 1;
        let _ = RollbackScope::run(&mut value, |scope, ctx| {
            let previous = *ctx;
            scope.defer(move |ctx| *ctx = previous);
            *ctx = 2;
            Err::<(), _>("fail")
        });
        assert_eq!(value, 1);
    }

    #[test]
    fn undoes_lifo() {
        let mut value = vec![1];
        let _ = RollbackScope::run(&mut value, |scope, ctx| {
            ctx.push(2);
            scope.defer(|ctx| {
                ctx.pop();
            });
            ctx.push(3);
            scope.defer(|ctx| {
                ctx.pop();
            });
            Err::<(), ()>(())
        });
        assert_eq!(value, vec![1]);
    }
}
